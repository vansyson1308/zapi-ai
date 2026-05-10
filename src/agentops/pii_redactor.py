"""
PIIRedactor — fast, dependency-free PII detection + redaction.

Scope:
- Pre-prompt redaction (apply to ChatCompletionRequest.messages before sending
  to a provider).
- Tool output redaction (apply to results returned from MCP/tools).

Trade-offs:
- We use regex-based detection for the common, high-precision categories
  (EMAIL, PHONE, IPV4, CREDIT_CARD, SSN, AWS_ACCESS_KEY). For broader/named-
  entity detection, callers can plug in presidio or a custom NER model via
  `register_pattern`.
- We deliberately avoid heavy NLP dependencies in the default path so this
  can run in the request hot path without latency surprises.

False positives:
- Phone-number regex is the loosest of the bunch; we keep it conservative
  (requires either separators or country-code prefix).
- Credit-card regex requires Luhn-valid digit sequences.

Usage:

    redactor = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    cleaned, matches = redactor.scan_and_apply(text)

    # In BLOCK mode:
    redactor.scan_or_raise(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

from .errors import PIIDetectedError
from .models import PIIPolicy, PIIPolicyMode


@dataclass(frozen=True)
class PIIMatch:
    """A single PII detection."""

    label: str
    start: int
    end: int
    raw: str


# ============================================================
# Default patterns
# ============================================================


DEFAULT_PATTERNS: Dict[str, Pattern[str]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    # Phone: requires at least one separator or +, 10-15 digits, avoids matching
    # things like timestamps (e.g. "1234567890123456" doesn't match — credit-card-like).
    "PHONE": re.compile(
        r"(?:(?:\+?\d{1,3}[\s.\-]?)?\(?\d{2,4}\)?[\s.\-]\d{2,4}[\s.\-]\d{2,5}(?:[\s.\-]\d{1,5})?)"
    ),
    "IPV4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
    # SSN: ###-##-####, with reasonable bounds
    "SSN": re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"),
    # Credit card: 13-19 digits, with optional separators. We Luhn-check below.
    "CREDIT_CARD": re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
    # AWS access key id: AKIA + 16 uppercase alphanumerics
    "AWS_ACCESS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # JWT: three base64url segments separated by dots. Accept a leading "eyJ"
    # to dodge most false positives on words.
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
}


# ============================================================
# Helpers
# ============================================================


def _luhn_valid(digits: str) -> bool:
    """Return True if the digit sequence passes Luhn checksum."""
    s = [int(c) for c in digits if c.isdigit()]
    if len(s) < 13 or len(s) > 19:
        return False
    checksum = 0
    parity = len(s) % 2
    for i, d in enumerate(s):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ============================================================
# Redactor
# ============================================================


class PIIRedactor:
    """Detect + redact PII from arbitrary text given a PIIPolicy."""

    def __init__(
        self,
        policy: Optional[PIIPolicy] = None,
        patterns: Optional[Dict[str, Pattern[str]]] = None,
    ) -> None:
        self.policy = policy or PIIPolicy(mode=PIIPolicyMode.REDACT)
        # Take a copy so callers can't mutate our class-level defaults
        self._patterns: Dict[str, Pattern[str]] = dict(patterns or DEFAULT_PATTERNS)

    # ------------------------------------------------------------------
    # pattern registry
    # ------------------------------------------------------------------

    def register_pattern(self, label: str, pattern: Pattern[str]) -> None:
        """Add or replace a pattern. Label SHOULD be uppercase."""
        self._patterns[label] = pattern

    def remove_pattern(self, label: str) -> bool:
        """Remove a pattern. Returns True if it existed."""
        return self._patterns.pop(label, None) is not None

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------

    def scan(self, text: str) -> List[PIIMatch]:
        """Return all matches found, deduplicated by (start, end)."""
        if not text or self.policy.mode is PIIPolicyMode.OFF:
            return []

        results: List[PIIMatch] = []
        seen: set = set()
        for label, regex in self._patterns.items():
            if not self._is_label_active(label):
                continue
            for m in regex.finditer(text):
                # Credit-card secondary check (Luhn)
                raw = m.group(0)
                if label == "CREDIT_CARD" and not _luhn_valid(raw):
                    continue
                key = (m.start(), m.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                results.append(PIIMatch(label=label, start=m.start(), end=m.end(), raw=raw))

        # Sort so substitution order is deterministic (left-to-right)
        results.sort(key=lambda x: (x.start, x.end))
        return results

    def _is_label_active(self, label: str) -> bool:
        if self.policy.allow_labels and label in self.policy.allow_labels:
            return False
        if self.policy.enforce_labels and label not in self.policy.enforce_labels:
            return False
        return True

    # ------------------------------------------------------------------
    # apply policy
    # ------------------------------------------------------------------

    def scan_and_apply(
        self,
        text: str,
        request_id: str = "",
    ) -> Tuple[str, List[PIIMatch]]:
        """
        Apply the policy mode to `text`.

        Returns (possibly modified text, list of matches).

        Raises PIIDetectedError if mode=BLOCK and at least one PII is found.
        """
        matches = self.scan(text)

        if self.policy.mode is PIIPolicyMode.OFF or not matches:
            return text, matches

        if self.policy.mode is PIIPolicyMode.DETECT:
            return text, matches

        if self.policy.mode is PIIPolicyMode.BLOCK:
            raise PIIDetectedError(labels=[m.label for m in matches], request_id=request_id)

        # REDACT: rebuild the string with placeholders
        return self._redact(text, matches), matches

    def scan_or_raise(self, text: str, request_id: str = "") -> List[PIIMatch]:
        """Lightweight alternative — only check, never mutate."""
        matches = self.scan(text)
        if matches and self.policy.mode is PIIPolicyMode.BLOCK:
            raise PIIDetectedError(labels=[m.label for m in matches], request_id=request_id)
        return matches

    @staticmethod
    def _redact(text: str, matches: List[PIIMatch]) -> str:
        if not matches:
            return text
        # Walk left-to-right; matches are pre-sorted.
        out: List[str] = []
        cursor = 0
        for m in matches:
            if m.start < cursor:
                # overlapping match (shouldn't happen with our regex set, but be safe)
                continue
            out.append(text[cursor:m.start])
            out.append(f"[REDACTED:{m.label}]")
            cursor = m.end
        out.append(text[cursor:])
        return "".join(out)

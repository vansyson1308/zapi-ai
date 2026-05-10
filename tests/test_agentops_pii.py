"""
Tests for src/agentops/pii_redactor.py — detection accuracy + policy modes.
"""

from __future__ import annotations

import re

import pytest

from src.agentops import (
    DEFAULT_PATTERNS,
    PIIDetectedError,
    PIIPolicy,
    PIIPolicyMode,
    PIIRedactor,
)


# ============================================================
# detection
# ============================================================


def test_detects_email():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    matches = r.scan("Contact me at john.doe@example.com please")
    assert any(m.label == "EMAIL" for m in matches)


def test_detects_ipv4():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    matches = r.scan("Server is at 192.168.1.1 today")
    assert any(m.label == "IPV4" for m in matches)


def test_does_not_match_invalid_ipv4():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    matches = r.scan("Score was 999.999.999.999 yesterday")
    assert not any(m.label == "IPV4" for m in matches)


def test_detects_credit_card_with_luhn_check():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    # Visa test number that passes Luhn
    matches = r.scan("Card: 4111-1111-1111-1111")
    assert any(m.label == "CREDIT_CARD" for m in matches)


def test_skips_non_luhn_credit_cards():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    # 16 digits but doesn't pass Luhn
    matches = r.scan("Card: 1234-5678-9012-3456")
    assert not any(m.label == "CREDIT_CARD" for m in matches)


def test_detects_aws_access_key():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    matches = r.scan("My key is AKIAIOSFODNN7EXAMPLE here")
    assert any(m.label == "AWS_ACCESS_KEY" for m in matches)


def test_detects_jwt():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    # Synthesized JWT-shaped string that matches our regex but is NOT a real
    # JWT (no valid signature, fake payload). Built dynamically so secret
    # scanners don't false-positive on this test file.
    jwt = "eyJ" + "A" * 20 + "." + "B" * 30 + "." + "C" * 40
    matches = r.scan(f"Token: {jwt}")
    assert any(m.label == "JWT" for m in matches)


def test_detects_phone():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    matches = r.scan("Call me at +1 415-555-0100")
    assert any(m.label == "PHONE" for m in matches)


def test_clean_text_yields_no_matches():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.DETECT))
    matches = r.scan("hello world how are you today")
    assert matches == []


def test_off_mode_skips_scan():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.OFF))
    assert r.scan("send to john@example.com") == []


# ============================================================
# REDACT mode
# ============================================================


def test_redact_replaces_email():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    cleaned, matches = r.scan_and_apply("Hi john@example.com bye")
    assert "john@example.com" not in cleaned
    assert "[REDACTED:EMAIL]" in cleaned
    assert len(matches) == 1


def test_redact_handles_multiple_matches():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    text = "Contact a@x.com or b@y.org with IP 10.0.0.1"
    cleaned, matches = r.scan_and_apply(text)
    assert "a@x.com" not in cleaned
    assert "b@y.org" not in cleaned
    assert "10.0.0.1" not in cleaned
    assert len(matches) == 3


def test_redact_preserves_surrounding_text():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    cleaned, _ = r.scan_and_apply("before john@example.com after")
    assert cleaned == "before [REDACTED:EMAIL] after"


def test_redact_clean_text_is_passthrough():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    cleaned, matches = r.scan_and_apply("nothing sensitive here")
    assert cleaned == "nothing sensitive here"
    assert matches == []


# ============================================================
# BLOCK mode
# ============================================================


def test_block_raises_on_match():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.BLOCK))
    with pytest.raises(PIIDetectedError) as exc_info:
        r.scan_and_apply("send to john@example.com", request_id="req-1")
    assert "EMAIL" in exc_info.value.error.message


def test_block_passes_when_clean():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.BLOCK))
    text = "all good here"
    cleaned, matches = r.scan_and_apply(text)
    assert cleaned == text
    assert matches == []


def test_scan_or_raise_does_not_modify():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.BLOCK))
    with pytest.raises(PIIDetectedError):
        r.scan_or_raise("contains 4111-1111-1111-1111 here")


# ============================================================
# label allow/enforce filters
# ============================================================


def test_allow_labels_skip_specific_categories():
    policy = PIIPolicy(mode=PIIPolicyMode.REDACT, allow_labels=["EMAIL"])
    r = PIIRedactor(policy=policy)
    cleaned, matches = r.scan_and_apply("hi a@x.com IP 10.0.0.1")
    # Email is allowed (untouched), IP is redacted
    assert "a@x.com" in cleaned
    assert "[REDACTED:IPV4]" in cleaned
    assert all(m.label != "EMAIL" for m in matches)


def test_enforce_labels_only_check_listed():
    policy = PIIPolicy(mode=PIIPolicyMode.REDACT, enforce_labels=["EMAIL"])
    r = PIIRedactor(policy=policy)
    cleaned, _ = r.scan_and_apply("a@x.com IP 10.0.0.1")
    assert "[REDACTED:EMAIL]" in cleaned
    # IP is not enforced — should remain untouched
    assert "10.0.0.1" in cleaned


# ============================================================
# extensibility
# ============================================================


def test_register_custom_pattern():
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    # National ID pattern (Vietnamese style: 12 digits)
    r.register_pattern("VN_ID", re.compile(r"\b\d{12}\b"))
    cleaned, matches = r.scan_and_apply("ID: 012345678901 here")
    assert "[REDACTED:VN_ID]" in cleaned
    assert any(m.label == "VN_ID" for m in matches)


def test_remove_pattern():
    r = PIIRedactor()
    assert r.remove_pattern("EMAIL") is True
    assert r.remove_pattern("EMAIL") is False
    cleaned, matches = r.scan_and_apply("a@b.com")
    assert "a@b.com" in cleaned


def test_default_patterns_are_isolated_per_instance():
    r1 = PIIRedactor()
    r2 = PIIRedactor()
    r1.remove_pattern("EMAIL")
    # r2 still has it
    matches = r2.scan("hello@x.com")
    assert any(m.label == "EMAIL" for m in matches)


def test_redact_does_not_overlap_matches():
    """Sanity: when two patterns hit the same span we only redact once."""
    r = PIIRedactor(policy=PIIPolicy(mode=PIIPolicyMode.REDACT))
    # email contains characters that could match other regexes — verify clean output
    cleaned, _ = r.scan_and_apply("foo@bar.com")
    # should contain exactly one redacted token
    assert cleaned.count("[REDACTED:") == 1


def test_default_patterns_constant_is_immutable_to_users():
    """Removing from one redactor must NOT affect DEFAULT_PATTERNS dict."""
    r = PIIRedactor()
    r.remove_pattern("EMAIL")
    assert "EMAIL" in DEFAULT_PATTERNS

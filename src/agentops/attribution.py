"""
Attribution helpers — extract per-customer/feature/agent identifiers from
incoming requests, and extend usage records with them so the Admin UI / SQL
analytics layer can answer "which customer racked up this cost?".

Headers we read (case-insensitive, since FastAPI normalizes them):
    X-Customer-Id    — opaque ID for the customer that made the call
    X-Feature-Id     — opaque ID for the feature/product surface
    X-Agent-Id       — opaque ID for the agent/run
    X-Session-Id     — opaque session correlator (optional)
    X-Attr-*         — any extra dimension; key is the suffix lowercased

Validation:
- We trim whitespace and reject ids longer than 256 chars (DoS guard).
- We never log the raw values at INFO level — they may contain user PII.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Mapping, Optional

from .models import AttributionContext

MAX_ID_LENGTH = 256
SAFE_ID_REGEX = re.compile(r"^[A-Za-z0-9._:\-]+$")


def _clean(value: Optional[str]) -> Optional[str]:
    """Trim, length-limit, and discard obviously hostile characters.

    Also normalizes Unicode to NFC so visually-identical strings encoded
    differently (precomposed vs decomposed combining characters) don't end
    up as separate Redis keys / attribution buckets.
    """
    if value is None:
        return None
    # NFC normalization first — collapses combining-character permutations.
    v = unicodedata.normalize("NFC", value).strip()
    if not v:
        return None
    if len(v) > MAX_ID_LENGTH:
        v = v[:MAX_ID_LENGTH]
    if not SAFE_ID_REGEX.match(v):
        # Strip control chars / spaces / unsafe characters; keep simple.
        v = re.sub(r"[^A-Za-z0-9._:\-]", "_", v)
        if not v:
            return None
    return v


def attribution_context_from_headers(
    headers: Mapping[str, str],
    *,
    tenant_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
) -> AttributionContext:
    """
    Build an AttributionContext from request headers.

    Header lookup is case-insensitive; we accept the common spellings users
    actually send (X-Customer-Id, x-customer-id).
    """
    # Normalize header lookup (FastAPI's Headers obj is already case-insensitive,
    # but we may also be called with a plain dict from tests).
    norm = {k.lower(): v for k, v in headers.items()}

    extra: Dict[str, str] = {}
    for key, value in norm.items():
        if key.startswith("x-attr-"):
            cleaned = _clean(value)
            if cleaned is not None:
                extra[key[len("x-attr-"):]] = cleaned

    return AttributionContext(
        tenant_id=_clean(tenant_id),
        api_key_id=_clean(api_key_id),
        customer_id=_clean(norm.get("x-customer-id")),
        feature_id=_clean(norm.get("x-feature-id")),
        agent_id=_clean(norm.get("x-agent-id")),
        session_id=_clean(norm.get("x-session-id")),
        extra=extra,
    )


def extend_usage_record(
    record_metadata: Dict[str, Any],
    attribution: AttributionContext,
) -> Dict[str, Any]:
    """
    Mutates and returns the metadata dict of a UsageRecord to embed
    attribution dimensions. We store under a top-level "attribution" key so
    SQL queries can `metadata->'attribution'->>'customer_id'` cleanly.

    Returns the same dict for chaining.
    """
    if attribution.is_empty():
        return record_metadata
    record_metadata["attribution"] = attribution.to_dict()
    return record_metadata

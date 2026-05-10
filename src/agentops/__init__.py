"""
AgentOps — Control plane for production AI agents.

Modules:
    guardian       — kill switches + budget caps
    attribution    — per-customer/feature/agent cost attribution
    pii_redactor   — PII detection + redaction in prompts and tool outputs
    middleware     — FastAPI middleware that ties them together
    errors         — agentops-specific exceptions
    models         — shared dataclasses

Public API is intentionally narrow. Most users only need:

    from src.agentops import Guardian, AttributionContext, BudgetCap
    from src.agentops import GuardianMiddleware
"""

from .errors import (
    AgentOpsError,
    BudgetExceededError,
    BudgetMidStreamExceededError,
    PIIDetectedError,
)
from .models import (
    AttributionContext,
    BudgetCap,
    BudgetPeriod,
    BudgetScope,
    BudgetUsageSnapshot,
    HardAction,
    PIIPolicy,
    PIIPolicyMode,
)
from .guardian import Guardian
from .attribution import attribution_context_from_headers, extend_usage_record
from .pii_redactor import (
    PIIRedactor,
    PIIMatch,
    DEFAULT_PATTERNS,
)

__all__ = [
    # errors
    "AgentOpsError",
    "BudgetExceededError",
    "BudgetMidStreamExceededError",
    "PIIDetectedError",
    # models
    "AttributionContext",
    "BudgetCap",
    "BudgetPeriod",
    "BudgetScope",
    "BudgetUsageSnapshot",
    "HardAction",
    "PIIPolicy",
    "PIIPolicyMode",
    # core
    "Guardian",
    "PIIRedactor",
    "PIIMatch",
    "DEFAULT_PATTERNS",
    # helpers
    "attribution_context_from_headers",
    "extend_usage_record",
]

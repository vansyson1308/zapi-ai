"""AgentOps-specific exceptions.

We deliberately raise SemanticError subclasses (not InfraError) — these
errors mean "the user/agent has hit a configured policy boundary and
should stop", not "infrastructure failed, retry might help".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import ErrorDetails, ErrorType, SemanticError


class AgentOpsError(SemanticError):
    """Base for AgentOps semantic errors."""

    def __init__(
        self,
        code: str,
        message: str,
        request_id: str = "",
        status_code: int = 402,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = ErrorDetails(
            code=code,
            message=message,
            type=ErrorType.SEMANTIC,
            request_id=request_id,
            retryable=False,
        )
        super().__init__(details, status_code=status_code)
        self.extra = extra or {}


class BudgetExceededError(AgentOpsError):
    """Raised pre-request when a budget cap would be breached by this call."""

    def __init__(
        self,
        scope: str,
        scope_id: str,
        period: str,
        limit_usd: Optional[float] = None,
        current_usd: Optional[float] = None,
        request_id: str = "",
    ) -> None:
        message = (
            f"Budget cap exceeded for {scope}={scope_id} period={period}: "
            f"current={current_usd} limit={limit_usd}"
        )
        super().__init__(
            code="agentops_budget_exceeded",
            message=message,
            request_id=request_id,
            status_code=402,
            extra={
                "scope": scope,
                "scope_id": scope_id,
                "period": period,
                "limit_usd": limit_usd,
                "current_usd": current_usd,
            },
        )


class BudgetMidStreamExceededError(AgentOpsError):
    """Raised mid-stream when a budget cap is breached during generation."""

    def __init__(
        self,
        scope: str,
        scope_id: str,
        period: str,
        partial_tokens: int = 0,
        request_id: str = "",
    ) -> None:
        message = (
            f"Budget cap exceeded mid-stream for {scope}={scope_id} period={period}; "
            f"halted after partial_tokens={partial_tokens}"
        )
        super().__init__(
            code="agentops_budget_exceeded_midstream",
            message=message,
            request_id=request_id,
            status_code=402,
            extra={
                "scope": scope,
                "scope_id": scope_id,
                "period": period,
                "partial_tokens": partial_tokens,
            },
        )


class PIIDetectedError(AgentOpsError):
    """Raised when PII is detected and policy is set to BLOCK."""

    def __init__(
        self,
        labels: List[str],
        request_id: str = "",
    ) -> None:
        message = f"PII detected and blocked: {labels}"
        super().__init__(
            code="agentops_pii_blocked",
            message=message,
            request_id=request_id,
            status_code=400,
            extra={"labels": list(labels)},
        )

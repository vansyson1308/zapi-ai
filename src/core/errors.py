"""
2api.ai - Error Definitions

Comprehensive error taxonomy with infra vs semantic classification.
See docs/ERROR_TAXONOMY.md for full specification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(str, Enum):
    """Error classification."""
    INFRA = "infra_error"
    SEMANTIC = "semantic_error"


@dataclass
class ErrorDetails:
    """Full error information for API response."""
    # Core fields (always present)
    code: str
    message: str
    type: ErrorType
    
    # Context fields
    provider: Optional[str] = None
    param: Optional[str] = None
    
    # Trace fields
    request_id: str = ""
    provider_request_id: Optional[str] = None
    
    # Recovery fields
    retryable: bool = False
    retry_after: Optional[int] = None
    fallback_attempted: Optional[bool] = None
    partial_content: Optional[str] = None
    
    # Debug fields
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "code": self.code,
            "message": self.message,
            "type": self.type.value,
            "request_id": self.request_id,
            "retryable": self.retryable,
        }
        
        if self.provider:
            result["provider"] = self.provider
        if self.param:
            result["param"] = self.param
        if self.provider_request_id:
            result["provider_request_id"] = self.provider_request_id
        if self.retry_after is not None:
            result["retry_after"] = self.retry_after
        if self.fallback_attempted is not None:
            result["fallback_attempted"] = self.fallback_attempted
        if self.partial_content:
            result["partial_content"] = self.partial_content
        if self.details:
            result["details"] = self.details
        
        return {"error": result}


class TwoApiException(Exception):
    """Base exception for all 2api.ai errors."""
    
    def __init__(self, error: ErrorDetails, status_code: int = 500):
        self.error = error
        self.status_code = status_code
        super().__init__(error.message)


# ============================================================
# Infra Errors (Retryable)
# ============================================================

class InfraError(TwoApiException):
    """Base class for infrastructure errors."""
    pass


class ConnectionTimeoutError(InfraError):
    """Failed to connect to provider."""
    
    def __init__(self, provider: str, request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="connection_timeout",
                message=f"Failed to connect to {provider} API within timeout",
                type=ErrorType.INFRA,
                provider=provider,
                request_id=request_id,
                retryable=True,
                retry_after=5
            ),
            status_code=504
        )


class ReadTimeoutError(InfraError):
    """Provider did not respond in time."""
    
    def __init__(self, provider: str, request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="read_timeout",
                message=f"{provider} did not respond within timeout",
                type=ErrorType.INFRA,
                provider=provider,
                request_id=request_id,
                retryable=True,
                retry_after=10
            ),
            status_code=504
        )


class UpstreamError(InfraError):
    """Provider returned server error."""
    
    def __init__(
        self,
        provider: str,
        status_code: int,
        message: str = "",
        request_id: str = "",
        provider_request_id: str = ""
    ):
        code_map = {
            500: "upstream_500",
            502: "upstream_502",
            503: "upstream_503",
            504: "upstream_504",
        }
        super().__init__(
            ErrorDetails(
                code=code_map.get(status_code, "upstream_error"),
                message=message or f"{provider} returned error {status_code}",
                type=ErrorType.INFRA,
                provider=provider,
                request_id=request_id,
                provider_request_id=provider_request_id or None,
                retryable=True,
                retry_after=30
            ),
            status_code=502 if status_code == 500 else status_code
        )


class RateLimitedError(InfraError):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        provider: str,
        retry_after: int = 60,
        limit_type: str = "requests",
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="rate_limited",
                message=f"{provider} rate limit exceeded. Retry after {retry_after} seconds.",
                type=ErrorType.INFRA,
                provider=provider,
                request_id=request_id,
                retryable=True,
                retry_after=retry_after,
                details={"limit_type": limit_type}
            ),
            status_code=429
        )


class StreamInterruptedError(InfraError):
    """Stream was interrupted after content was produced."""
    
    def __init__(
        self,
        provider: str,
        partial_content: str = "",
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="stream_interrupted",
                message="Connection lost after receiving partial content",
                type=ErrorType.INFRA,
                provider=provider,
                request_id=request_id,
                retryable=False,  # Critical: No retry for semantic drift
                partial_content=partial_content
            ),
            status_code=502
        )


class ProviderDownError(InfraError):
    """Provider appears to be completely down."""
    
    def __init__(self, provider: str, request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="provider_down",
                message=f"{provider} appears to be unavailable",
                type=ErrorType.INFRA,
                provider=provider,
                request_id=request_id,
                retryable=True,
                fallback_attempted=True
            ),
            status_code=503
        )


class AllProvidersFailedError(InfraError):
    """All providers in fallback chain failed."""
    
    def __init__(self, providers: list, last_error: str = "", request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="all_providers_failed",
                message="All providers in fallback chain failed",
                type=ErrorType.INFRA,
                request_id=request_id,
                retryable=False,
                fallback_attempted=True,
                details={
                    "providers_tried": providers,
                    "last_error": last_error
                }
            ),
            status_code=503
        )


# ============================================================
# Semantic Errors (Not Retryable)
# ============================================================

class SemanticError(TwoApiException):
    """Base class for semantic errors (client must fix request)."""
    pass


class InvalidAPIKeyError(SemanticError):
    """API key is invalid."""
    
    def __init__(self, message: str = "Invalid API key", request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="invalid_api_key",
                message=message,
                type=ErrorType.SEMANTIC,
                request_id=request_id,
                retryable=False
            ),
            status_code=401
        )


class MissingAPIKeyError(SemanticError):
    """No API key provided."""
    
    def __init__(self, request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="missing_api_key",
                message="Authorization header required",
                type=ErrorType.SEMANTIC,
                request_id=request_id,
                retryable=False
            ),
            status_code=401
        )


class InvalidRequestError(SemanticError):
    """Request validation failed."""
    
    def __init__(
        self,
        message: str,
        param: str = "",
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="invalid_request",
                message=message,
                type=ErrorType.SEMANTIC,
                param=param or None,
                request_id=request_id,
                retryable=False
            ),
            status_code=400
        )


class MissingRequiredFieldError(SemanticError):
    """Required field is missing."""
    
    def __init__(self, field: str, request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="missing_required_field",
                message=f"'{field}' is required",
                type=ErrorType.SEMANTIC,
                param=field,
                request_id=request_id,
                retryable=False
            ),
            status_code=400
        )


class ModelNotFoundError(SemanticError):
    """Requested model does not exist."""
    
    def __init__(self, model: str, request_id: str = ""):
        super().__init__(
            ErrorDetails(
                code="model_not_found",
                message=f"Model '{model}' not found",
                type=ErrorType.SEMANTIC,
                request_id=request_id,
                retryable=False,
                details={"requested_model": model}
            ),
            status_code=404
        )


class ContentFilteredError(SemanticError):
    """Content was filtered by safety systems."""
    
    def __init__(
        self,
        provider: str,
        reason: str = "",
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="content_filtered",
                message="Your request was flagged by content moderation",
                type=ErrorType.SEMANTIC,
                provider=provider,
                request_id=request_id,
                retryable=False,
                details={"filter_reason": reason} if reason else {}
            ),
            status_code=400
        )


class ContextLengthExceededError(SemanticError):
    """Input exceeds model's context window."""
    
    def __init__(
        self,
        model: str,
        max_tokens: int,
        actual_tokens: int,
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="context_length_exceeded",
                message=f"Input ({actual_tokens} tokens) exceeds {model} context window ({max_tokens} tokens)",
                type=ErrorType.SEMANTIC,
                request_id=request_id,
                retryable=False,
                details={
                    "model": model,
                    "max_context_tokens": max_tokens,
                    "actual_tokens": actual_tokens
                }
            ),
            status_code=400
        )


class ToolSchemaInvalidError(SemanticError):
    """Tool schema is invalid."""
    
    def __init__(
        self,
        tool_name: str,
        reason: str,
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="tool_schema_invalid",
                message=f"Tool '{tool_name}' has invalid schema: {reason}",
                type=ErrorType.SEMANTIC,
                param=f"tools[].function",
                request_id=request_id,
                retryable=False,
                details={"tool_name": tool_name}
            ),
            status_code=400
        )


class ToolSchemaIncompatibleError(SemanticError):
    """Tool schema uses features not supported by target provider."""
    
    def __init__(
        self,
        tool_name: str,
        provider: str,
        unsupported_features: list,
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="tool_schema_incompatible",
                message=f"Tool '{tool_name}' uses features not supported by {provider}",
                type=ErrorType.SEMANTIC,
                provider=provider,
                param="tools[].function.parameters",
                request_id=request_id,
                retryable=False,
                details={
                    "tool_name": tool_name,
                    "unsupported_features": unsupported_features
                }
            ),
            status_code=400
        )


class BudgetExceededError(SemanticError):
    """Request would exceed tenant's budget."""
    
    def __init__(
        self,
        budget_limit: float,
        current_spending: float,
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code="budget_exceeded",
                message=f"Request would exceed monthly budget of ${budget_limit:.2f}",
                type=ErrorType.SEMANTIC,
                request_id=request_id,
                retryable=False,
                details={
                    "budget_limit_usd": budget_limit,
                    "current_spending_usd": current_spending
                }
            ),
            status_code=402
        )


class TenantRateLimitedError(SemanticError):
    """Tenant's own rate limit exceeded (not provider's)."""
    
    def __init__(
        self,
        limit_type: str,
        limit: int,
        retry_after: int = 60,
        request_id: str = ""
    ):
        super().__init__(
            ErrorDetails(
                code=f"tenant_{limit_type}_exceeded",
                message=f"Your rate limit ({limit} {limit_type}) exceeded",
                type=ErrorType.SEMANTIC,  # Semantic because it's tenant's own limit
                request_id=request_id,
                retryable=True,  # Can retry after cooldown
                retry_after=retry_after,
                details={"limit_type": limit_type, "limit": limit}
            ),
            status_code=429
        )


# ============================================================
# Error Factory
# ============================================================

def create_error_from_provider(
    provider: str,
    status_code: int,
    error_body: dict,
    request_id: str = "",
    provider_request_id: str = ""
) -> TwoApiException:
    """Create appropriate error from provider response."""
    
    message = error_body.get("message", str(error_body))
    error_type = error_body.get("type", "")
    error_code = error_body.get("code", "")
    
    # Map common provider errors
    if status_code == 401:
        return SemanticError(
            ErrorDetails(
                code="provider_auth_error",
                message=f"{provider} authentication failed",
                type=ErrorType.SEMANTIC,
                provider=provider,
                request_id=request_id,
                retryable=False
            ),
            status_code=502
        )
    
    if status_code == 429:
        retry_after = 60
        if "retry-after" in error_body:
            retry_after = int(error_body["retry-after"])
        return RateLimitedError(provider, retry_after, request_id=request_id)
    
    if status_code >= 500:
        return UpstreamError(
            provider, status_code, message,
            request_id, provider_request_id
        )
    
    if "content" in error_code.lower() or "safety" in error_type.lower():
        return ContentFilteredError(provider, error_type, request_id)
    
    if "context" in message.lower() or "token" in message.lower():
        return SemanticError(
            ErrorDetails(
                code="provider_context_error",
                message=message,
                type=ErrorType.SEMANTIC,
                provider=provider,
                request_id=request_id,
                retryable=False
            ),
            status_code=400
        )
    
    # Default: treat as semantic error from provider
    return SemanticError(
        ErrorDetails(
            code="provider_error",
            message=message,
            type=ErrorType.SEMANTIC,
            provider=provider,
            request_id=request_id,
            provider_request_id=provider_request_id or None,
            retryable=False
        ),
        status_code=status_code if status_code < 500 else 400
    )

"""
2api.ai Core Module

Contains the unified data models for all AI providers.
"""

from .models import (
    # Enums
    Provider,
    Role,
    FinishReason,
    RoutingStrategy,
    
    # Messages
    Message,
    ContentPart,
    TextContent,
    ImageContent,
    ImageUrl,
    
    # Tool calling
    Tool,
    ToolCall,
    FunctionDefinition,
    FunctionCall,
    
    # Requests
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    RoutingConfig,
    
    # Responses
    ChatCompletionResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    Choice,
    Usage,
    TwoApiMetadata,
    RoutingDecision,
    
    # Model info
    ModelInfo,
    ModelPricing,
    
    # Errors
    TwoApiError,
    InvalidRequestError,
    AuthenticationError,
    RateLimitError,
    ProviderError,
    APIError,
    
    # Serialization
    message_to_dict,
    response_to_dict,
)

from .errors import (
    # Error types
    ErrorType,
    ErrorDetails,
    TwoApiException,
    
    # Infra errors
    InfraError,
    ConnectionTimeoutError,
    ReadTimeoutError,
    UpstreamError,
    RateLimitedError,
    StreamInterruptedError,
    ProviderDownError,
    AllProvidersFailedError,
    
    # Semantic errors
    SemanticError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    InvalidRequestError as InvalidRequestErr,
    MissingRequiredFieldError,
    ModelNotFoundError,
    ContentFilteredError,
    ContextLengthExceededError,
    ToolSchemaInvalidError,
    ToolSchemaIncompatibleError,
    BudgetExceededError,
    TenantRateLimitedError,
    
    # Factory
    create_error_from_provider,
)

__all__ = [
    # Enums
    "Provider",
    "Role",
    "FinishReason",
    "RoutingStrategy",
    
    # Messages
    "Message",
    "ContentPart",
    "TextContent",
    "ImageContent",
    "ImageUrl",
    
    # Tool calling
    "Tool",
    "ToolCall",
    "FunctionDefinition",
    "FunctionCall",
    
    # Requests
    "ChatCompletionRequest",
    "EmbeddingRequest",
    "ImageGenerationRequest",
    "RoutingConfig",
    
    # Responses
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "ImageGenerationResponse",
    "Choice",
    "Usage",
    "TwoApiMetadata",
    "RoutingDecision",
    
    # Model info
    "ModelInfo",
    "ModelPricing",
    
    # Errors
    "TwoApiError",
    "InvalidRequestError",
    "AuthenticationError",
    "RateLimitError",
    "ProviderError",
    "APIError",
    
    # Serialization
    "message_to_dict",
    "response_to_dict",
]

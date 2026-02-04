"""
2api.ai - API Layer

Complete REST API implementation with OpenAI-compatible endpoints.

Provides:
- Chat completions (streaming and non-streaming)
- Embeddings
- Image generation
- Model listing and comparison
- Tenant/API key management
"""

from .management import router as management_router
from .models import (
    # Request models
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    # Response models
    ChatCompletionResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    ModelInfo,
    ModelListResponse,
    # Shared models
    MessageInput,
    RoleEnum,
    ToolDefinition,
    RoutingConfig,
    UsageInfo,
    # Additional models
    Choice,
    MessageOutput,
    ChatCompletionChunk,
)
from .dependencies import (
    get_router,
    add_standard_headers,
    check_rate_limits,
    start_request_tracking,
    RequestTimer,
)
from .middleware import (
    RequestLoggingMiddleware,
    MetricsMiddleware,
    RequestIdMiddleware,
    ErrorTrackingMiddleware,
)
from .routes import (
    chat_router,
    embeddings_router,
    images_router,
    models_router,
)


__all__ = [
    # Routers
    "management_router",
    "chat_router",
    "embeddings_router",
    "images_router",
    "models_router",
    # Request models
    "ChatCompletionRequest",
    "EmbeddingRequest",
    "ImageGenerationRequest",
    # Response models
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "ImageGenerationResponse",
    "ModelInfo",
    "ModelListResponse",
    # Shared models
    "MessageInput",
    "RoleEnum",
    "ToolDefinition",
    "RoutingConfig",
    "UsageInfo",
    "Choice",
    "MessageOutput",
    "ChatCompletionChunk",
    # Dependencies
    "get_router",
    "add_standard_headers",
    "check_rate_limits",
    "start_request_tracking",
    "RequestTimer",
    # Middleware
    "RequestLoggingMiddleware",
    "MetricsMiddleware",
    "RequestIdMiddleware",
    "ErrorTrackingMiddleware",
]

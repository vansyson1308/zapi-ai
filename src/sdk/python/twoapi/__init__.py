"""
2api.ai Python SDK

A simple, unified interface to access multiple AI providers.

Quick Start:
    from twoapi import TwoAPI

    client = TwoAPI(api_key="2api_xxx")

    # Simple chat
    response = client.chat("Hello!")
    print(response.content)

    # With specific model
    response = client.chat(
        "Explain quantum computing",
        model="anthropic/claude-3-5-sonnet"
    )

    # With streaming
    for chunk in client.chat_stream("Tell me a story"):
        print(chunk, end="", flush=True)

    # Async usage
    async_client = AsyncTwoAPI(api_key="2api_xxx")
    response = await async_client.chat("Hello!")
"""

from .client import TwoAPI
from .async_client import AsyncTwoAPI
from .models import (
    Message,
    Usage,
    RoutingInfo,
    ChatResponse,
    EmbeddingResponse,
    ImageResponse,
    ModelInfo,
    HealthStatus,
    ToolCall,
    Tool,
    FunctionDefinition,
)
from .errors import (
    TwoAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderError,
    TimeoutError,
    ConnectionError,
    StreamError,
    is_retryable_error,
)
from .utils import chat, chat_stream, embed, generate_image
from .tools import (
    ToolRunner,
    ToolFunction,
    ToolResult,
    RunResult,
    tool,
    create_tool,
)
from .models import RoutingConfig, RetryConfig

__version__ = "1.0.0"
__all__ = [
    # Clients
    "TwoAPI",
    "AsyncTwoAPI",
    # Models
    "Message",
    "Usage",
    "RoutingInfo",
    "ChatResponse",
    "EmbeddingResponse",
    "ImageResponse",
    "ModelInfo",
    "HealthStatus",
    "ToolCall",
    "Tool",
    "FunctionDefinition",
    "RoutingConfig",
    "RetryConfig",
    # Errors
    "TwoAPIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "ProviderError",
    "TimeoutError",
    "ConnectionError",
    "StreamError",
    "is_retryable_error",
    # Tool calling helpers
    "ToolRunner",
    "ToolFunction",
    "ToolResult",
    "RunResult",
    "tool",
    "create_tool",
    # Convenience functions
    "chat",
    "chat_stream",
    "embed",
    "generate_image",
]

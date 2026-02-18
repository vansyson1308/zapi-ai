"""
2api.ai - API Request/Response Models

Pydantic models for API validation and serialization.
These are the external-facing models that clients interact with.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================
# Enums
# ============================================================

class RoleEnum(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReasonEnum(str, Enum):
    """Reasons for completion finish."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


# ============================================================
# Content Parts (for multimodal messages)
# ============================================================

class TextContentPart(BaseModel):
    """Text content part."""
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    """Image URL reference."""
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContentPart(BaseModel):
    """Image content part."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Union[TextContentPart, ImageContentPart]


# ============================================================
# Tool Definitions
# ============================================================

class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    additionalProperties: bool = False

    model_config = ConfigDict(extra="allow")


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(default="", max_length=1024)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    strict: bool = False


class ToolDefinition(BaseModel):
    """Tool definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """Function call in tool call."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call from assistant."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ToolChoiceFunction(BaseModel):
    """Specific function choice."""
    name: str


class ToolChoiceObject(BaseModel):
    """Tool choice object for specific function."""
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


ToolChoice = Union[Literal["auto", "none", "required"], ToolChoiceObject]


# ============================================================
# Message Models
# ============================================================

class MessageInput(BaseModel):
    """
    Input message for chat completion.

    Supports:
    - Simple text messages
    - Multimodal messages (text + images)
    - Tool call messages
    - Tool result messages
    """
    role: RoleEnum
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = Field(default=None, max_length=64)
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @model_validator(mode="after")
    def validate_message(self):
        """Validate message based on role."""
        if self.role == RoleEnum.TOOL and not self.tool_call_id:
            raise ValueError("tool_call_id is required for tool messages")
        if self.role == RoleEnum.ASSISTANT and self.tool_calls and self.content:
            pass  # Both are allowed for assistant
        return self


class MessageOutput(BaseModel):
    """Output message in response."""
    role: RoleEnum
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    refusal: Optional[str] = None  # For content filter


# ============================================================
# Routing Configuration
# ============================================================

class RoutingConfig(BaseModel):
    """Routing configuration for requests."""
    strategy: Optional[Literal["cost", "latency", "quality", "balanced"]] = None
    fallback: Optional[List[str]] = Field(
        default=None,
        description="List of fallback providers (e.g., ['anthropic', 'google'])"
    )
    max_latency_ms: Optional[int] = Field(
        default=None,
        ge=100,
        le=300000,
        description="Maximum acceptable latency in milliseconds"
    )
    max_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum cost per request in USD"
    )


# ============================================================
# Chat Completion Request/Response
# ============================================================

class ChatCompletionRequest(BaseModel):
    """
    Chat completion request.

    Compatible with OpenAI's API format with 2api.ai extensions.
    """
    model: str = Field(
        ...,
        description="Model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet')"
    )
    messages: List[MessageInput] = Field(
        ...,
        min_length=1,
        description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=128000,
        description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    tools: Optional[List[ToolDefinition]] = Field(
        default=None,
        description="Available tools for the model"
    )
    tool_choice: Optional[ToolChoice] = Field(
        default=None,
        description="How the model should use tools"
    )

    # 2api.ai extensions
    routing: Optional[RoutingConfig] = Field(
        default=None,
        description="Routing configuration"
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom metadata for tracking"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Validate messages are not empty."""
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class Choice(BaseModel):
    """A single completion choice."""
    index: int
    message: MessageOutput
    finish_reason: Optional[FinishReasonEnum] = None
    logprobs: Optional[Any] = None


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TwoApiMetadata(BaseModel):
    """2api.ai specific metadata."""
    request_id: str
    latency_ms: int
    cost_usd: float
    provider: str
    routing_strategy: Optional[str] = None
    fallback_used: bool = False


class ChatCompletionResponse(BaseModel):
    """
    Chat completion response.

    Compatible with OpenAI's API format with 2api.ai metadata.
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None

    # 2api.ai extension
    _2api: Optional[TwoApiMetadata] = None


# ============================================================
# Streaming Models
# ============================================================

class DeltaMessage(BaseModel):
    """Delta message in streaming chunk."""
    role: Optional[RoleEnum] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class StreamChoice(BaseModel):
    """Choice in streaming response."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinishReasonEnum] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None  # Only in final chunk


# ============================================================
# Embedding Request/Response
# ============================================================

class EmbeddingRequest(BaseModel):
    """Embedding request."""
    model: str = Field(
        ...,
        description="Embedding model ID"
    )
    input: Union[str, List[str]] = Field(
        ...,
        description="Text to embed"
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Output encoding format"
    )
    dimensions: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Output dimensions (for models that support it)"
    )


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo


# ============================================================
# Image Generation Request/Response
# ============================================================

class ImageGenerationRequest(BaseModel):
    """Image generation request."""
    model: str = Field(
        default="dall-e-3",
        description="Image model (dall-e-3)"
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Image description"
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images"
    )
    size: Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"] = Field(
        default="1024x1024",
        description="Image size"
    )
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description="Image quality"
    )
    style: Literal["vivid", "natural"] = Field(
        default="vivid",
        description="Image style"
    )
    response_format: Literal["url", "b64_json"] = Field(
        default="url",
        description="Response format"
    )


class ImageData(BaseModel):
    """Single image result."""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """Image generation response."""
    created: int
    data: List[ImageData]


# ============================================================
# Model Information
# ============================================================

class ModelPricing(BaseModel):
    """Model pricing information."""
    input_per_1m_tokens: float
    output_per_1m_tokens: float


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: Literal["model"] = "model"
    provider: str
    name: str
    capabilities: List[str]
    context_window: int
    max_output_tokens: int
    pricing: ModelPricing
    created: Optional[int] = None


class ModelListResponse(BaseModel):
    """Model list response."""
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ============================================================
# Error Models
# ============================================================

class ErrorDetail(BaseModel):
    """Error detail."""
    code: str
    message: str
    type: str
    param: Optional[str] = None
    provider: Optional[str] = None
    request_id: Optional[str] = None
    retryable: bool = False
    retry_after: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: ErrorDetail


# ============================================================
# Health Check
# ============================================================

class ProviderHealth(BaseModel):
    """Provider health status."""
    status: Literal["healthy", "unhealthy", "degraded"]
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    providers: Dict[str, ProviderHealth]


# ============================================================
# Usage Statistics
# ============================================================

class UsageSummary(BaseModel):
    """Usage summary."""
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    by_provider: Dict[str, Dict[str, Any]]
    by_model: Dict[str, Dict[str, Any]]
    period: Dict[str, Optional[str]]


class UsageResponse(BaseModel):
    """Usage response."""
    object: Literal["usage_summary"] = "usage_summary"
    tenant_id: Optional[str] = None
    summary: UsageSummary

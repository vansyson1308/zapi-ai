"""
2api.ai - Core Data Models

Unified data models for all AI providers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


# ============================================================
# Enums
# ============================================================

class Provider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class Role(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Completion finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class RoutingStrategy(str, Enum):
    """Routing optimization strategies."""
    COST = "cost"
    LATENCY = "latency"
    QUALITY = "quality"


class ContentType(str, Enum):
    """Content part types."""
    TEXT = "text"
    IMAGE_URL = "image_url"


# ============================================================
# Content Parts (for multimodal)
# ============================================================

@dataclass
class ImageUrl:
    """Image URL for vision models."""
    url: str
    detail: Literal["low", "high", "auto"] = "auto"


@dataclass
class TextContent:
    """Text content part."""
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content part."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl = field(default_factory=lambda: ImageUrl(""))


ContentPart = Union[TextContent, ImageContent]


# ============================================================
# Tool Calling
# ============================================================

@dataclass
class FunctionDefinition:
    """Function definition for tool calling."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """Tool definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition = field(default_factory=lambda: FunctionDefinition(""))


@dataclass
class FunctionCall:
    """Function call made by the model."""
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    """Tool call in response."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall = field(default_factory=lambda: FunctionCall("", ""))


@dataclass
class ToolChoiceFunction:
    """Specific function to call."""
    name: str


@dataclass
class ToolChoiceSpecific:
    """Force specific tool."""
    type: Literal["function"] = "function"
    function: ToolChoiceFunction = field(default_factory=lambda: ToolChoiceFunction(""))


ToolChoice = Union[Literal["auto", "none", "required"], ToolChoiceSpecific]


# ============================================================
# Messages
# ============================================================

@dataclass
class Message:
    """
    Unified message format.
    
    Supports:
    - Simple text messages
    - Multimodal messages (text + images)
    - Tool call messages
    - Tool result messages
    """
    role: Role
    content: Union[str, List[ContentPart], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: Union[str, List[ContentPart]]) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> Message:
        """Create a tool result message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


# ============================================================
# Routing Configuration
# ============================================================

@dataclass
class RoutingConfig:
    """Configuration for intelligent routing."""
    strategy: Optional[RoutingStrategy] = None
    fallback: Optional[List[str]] = None
    max_latency_ms: Optional[int] = None
    max_cost: Optional[float] = None


# ============================================================
# Request Models
# ============================================================

@dataclass
class ChatCompletionRequest:
    """
    Unified chat completion request.
    
    Example:
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[
                Message.system("You are helpful."),
                Message.user("Hello!")
            ],
            temperature=0.7
        )
    """
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    routing: Optional[RoutingConfig] = None
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self):
        # Parse model string to extract provider
        if "/" in self.model:
            self._provider, self._model_name = self.model.split("/", 1)
        else:
            self._provider = None
            self._model_name = self.model

    @property
    def provider(self) -> Optional[str]:
        """Extract provider from model string."""
        return self._provider

    @property
    def model_name(self) -> str:
        """Extract model name from model string."""
        return self._model_name

    @property
    def is_auto_routing(self) -> bool:
        """Check if auto routing is requested."""
        return self.model.lower() == "auto"


@dataclass
class EmbeddingRequest:
    """Unified embedding request."""
    model: str
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None


@dataclass
class ImageGenerationRequest:
    """Unified image generation request."""
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: Literal["standard", "hd"] = "standard"
    style: Literal["vivid", "natural"] = "vivid"
    response_format: Literal["url", "b64_json"] = "url"


# ============================================================
# Response Models
# ============================================================

@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class RoutingDecision:
    """Information about routing decision."""
    strategy_used: str
    candidates_evaluated: List[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class TwoApiMetadata:
    """2api.ai specific metadata in response."""
    request_id: str
    latency_ms: int
    cost_usd: float
    routing_decision: Optional[RoutingDecision] = None


@dataclass
class Choice:
    """A single completion choice."""
    index: int
    message: Message
    finish_reason: FinishReason


@dataclass
class ChatCompletionResponse:
    """
    Unified chat completion response.
    
    Contains the actual response from the AI model plus
    metadata about the request (cost, latency, routing).
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    provider: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    _2api: Optional[TwoApiMetadata] = None

    @classmethod
    def create(
        cls,
        content: str,
        model: str,
        provider: str,
        usage: Usage,
        finish_reason: FinishReason = FinishReason.STOP,
        tool_calls: Optional[List[ToolCall]] = None,
        metadata: Optional[TwoApiMetadata] = None
    ) -> ChatCompletionResponse:
        """Helper to create a response."""
        return cls(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=model,
            provider=provider,
            choices=[
                Choice(
                    index=0,
                    message=Message.assistant(content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason
                )
            ],
            usage=usage,
            _2api=metadata
        )


@dataclass
class EmbeddingData:
    """Single embedding result."""
    object: Literal["embedding"] = "embedding"
    embedding: List[float] = field(default_factory=list)
    index: int = 0


@dataclass
class EmbeddingResponse:
    """Unified embedding response."""
    object: Literal["list"] = "list"
    data: List[EmbeddingData] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)


@dataclass
class ImageData:
    """Single image result."""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


@dataclass
class ImageGenerationResponse:
    """Unified image generation response."""
    created: int = field(default_factory=lambda: int(time.time()))
    data: List[ImageData] = field(default_factory=list)


# ============================================================
# Error Models
# ============================================================

@dataclass
class APIError:
    """Unified API error."""
    code: str
    message: str
    type: str
    provider: Optional[str] = None
    param: Optional[str] = None


class TwoApiError(Exception):
    """Base exception for 2api.ai errors."""
    
    def __init__(self, error: APIError):
        self.error = error
        super().__init__(error.message)


class InvalidRequestError(TwoApiError):
    """Invalid request parameters."""
    pass


class AuthenticationError(TwoApiError):
    """Authentication failed."""
    pass


class RateLimitError(TwoApiError):
    """Rate limit exceeded."""
    
    def __init__(self, error: APIError, retry_after: int = 60):
        super().__init__(error)
        self.retry_after = retry_after


class ProviderError(TwoApiError):
    """Error from AI provider."""
    pass


# ============================================================
# Model Information
# ============================================================

@dataclass
class ModelPricing:
    """Pricing information for a model."""
    input_per_1m_tokens: float
    output_per_1m_tokens: float


@dataclass
class ModelInfo:
    """Information about an AI model."""
    id: str
    provider: Provider
    name: str
    capabilities: List[str]
    context_window: int
    max_output_tokens: int
    pricing: ModelPricing

    def supports(self, capability: str) -> bool:
        """Check if model supports a capability."""
        return capability in self.capabilities

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage."""
        input_cost = (input_tokens / 1_000_000) * self.pricing.input_per_1m_tokens
        output_cost = (output_tokens / 1_000_000) * self.pricing.output_per_1m_tokens
        return input_cost + output_cost


# ============================================================
# Serialization Helpers
# ============================================================

def message_to_dict(msg: Message) -> Dict[str, Any]:
    """Convert Message to dictionary for JSON serialization."""
    result: Dict[str, Any] = {"role": msg.role.value}
    
    if msg.content is not None:
        if isinstance(msg.content, str):
            result["content"] = msg.content
        else:
            # Multimodal content
            result["content"] = [
                {
                    "type": part.type,
                    **({"text": part.text} if isinstance(part, TextContent) else {}),
                    **({"image_url": {"url": part.image_url.url, "detail": part.image_url.detail}}
                       if isinstance(part, ImageContent) else {})
                }
                for part in msg.content
            ]
    
    if msg.name:
        result["name"] = msg.name
    if msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in msg.tool_calls
        ]
    
    return result


def response_to_dict(resp: ChatCompletionResponse) -> Dict[str, Any]:
    """Convert ChatCompletionResponse to dictionary for JSON serialization."""
    result = {
        "id": resp.id,
        "object": resp.object,
        "created": resp.created,
        "model": resp.model,
        "provider": resp.provider,
        "choices": [
            {
                "index": c.index,
                "message": message_to_dict(c.message),
                "finish_reason": c.finish_reason.value
            }
            for c in resp.choices
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens
        }
    }
    
    if resp._2api:
        result["_2api"] = {
            "request_id": resp._2api.request_id,
            "latency_ms": resp._2api.latency_ms,
            "cost_usd": resp._2api.cost_usd,
        }
        if resp._2api.routing_decision:
            result["_2api"]["routing_decision"] = {
                "strategy_used": resp._2api.routing_decision.strategy_used,
                "candidates_evaluated": resp._2api.routing_decision.candidates_evaluated,
                "fallback_used": resp._2api.routing_decision.fallback_used
            }
    
    return result

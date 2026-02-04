"""
2api.ai SDK - Data Models

Pydantic-style dataclasses for type safety and convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ============================================================
# Message Models
# ============================================================

@dataclass
class FunctionDefinition:
    """Function definition for tool calling."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class Tool:
    """Tool definition for function calling."""
    function: FunctionDefinition
    type: str = "function"

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tool:
        """Create a tool with the given function definition."""
        return cls(
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}}
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "type": self.type,
            "function": self.function.to_dict()
        }


@dataclass
class FunctionCall:
    """Function call details."""
    name: str
    arguments: str  # JSON string

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    """Tool call from assistant."""
    id: str
    function: FunctionCall
    type: str = "function"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCall:
        """Create from dictionary."""
        func_data = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=FunctionCall(
                name=func_data.get("name", ""),
                arguments=func_data.get("arguments", "{}")
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict()
        }


@dataclass
class Message:
    """A chat message."""
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Union[str, List[Dict[str, Any]]]) -> Message:
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        """Create a tool result message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        return result


# ============================================================
# Response Models
# ============================================================

@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Usage:
        """Create from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0)
        )


@dataclass
class RoutingInfo:
    """Information about how the request was routed."""
    strategy_used: str = ""
    fallback_used: bool = False
    latency_ms: int = 0
    cost_usd: float = 0.0
    provider: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoutingInfo:
        """Create from 2api metadata."""
        routing_decision = data.get("routing_decision", {})
        return cls(
            strategy_used=routing_decision.get("strategy_used", ""),
            fallback_used=routing_decision.get("fallback_used", False),
            latency_ms=data.get("latency_ms", 0),
            cost_usd=data.get("cost_usd", 0.0),
            provider=data.get("provider", "")
        )


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: Optional[str]
    model: str
    provider: str
    usage: Usage
    id: str = ""
    finish_reason: str = "stop"
    tool_calls: Optional[List[ToolCall]] = None
    routing: Optional[RoutingInfo] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatResponse:
        """Create ChatResponse from API response dictionary."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {})
        twoapi_data = data.get("_2api", {})

        # Parse tool calls
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall.from_dict(tc)
                for tc in message["tool_calls"]
            ]

        # Parse routing info
        routing = None
        if twoapi_data:
            routing = RoutingInfo.from_dict(twoapi_data)

        return cls(
            id=data.get("id", ""),
            content=message.get("content"),
            model=data.get("model", ""),
            provider=data.get("provider", twoapi_data.get("provider", "")),
            usage=Usage.from_dict(usage_data),
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
            routing=routing
        )

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""
    embeddings: List[List[float]]
    model: str
    usage: Usage

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingResponse:
        """Create from API response."""
        usage_data = data.get("usage", {})
        return cls(
            embeddings=[item["embedding"] for item in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=0,
                total_tokens=usage_data.get("total_tokens", 0)
            )
        )


@dataclass
class ImageData:
    """Single image result."""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


@dataclass
class ImageResponse:
    """Response from an image generation request."""
    images: List[ImageData]
    created: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageResponse:
        """Create from API response."""
        items = data.get("data", [])
        return cls(
            images=[
                ImageData(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt")
                )
                for item in items
            ],
            created=data.get("created", 0)
        )

    @property
    def urls(self) -> List[str]:
        """Get list of image URLs."""
        return [img.url for img in self.images if img.url]

    @property
    def revised_prompts(self) -> List[str]:
        """Get list of revised prompts."""
        return [img.revised_prompt or "" for img in self.images]


@dataclass
class ModelPricing:
    """Model pricing information."""
    input_per_1m_tokens: float = 0.0
    output_per_1m_tokens: float = 0.0


@dataclass
class ModelInfo:
    """Model information."""
    id: str
    provider: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    context_window: int = 0
    max_output_tokens: int = 0
    pricing: ModelPricing = field(default_factory=ModelPricing)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelInfo:
        """Create from API response."""
        pricing_data = data.get("pricing", {})
        return cls(
            id=data.get("id", ""),
            provider=data.get("provider", ""),
            name=data.get("name", ""),
            capabilities=data.get("capabilities", []),
            context_window=data.get("context_window", 0),
            max_output_tokens=data.get("max_output_tokens", 0),
            pricing=ModelPricing(
                input_per_1m_tokens=pricing_data.get("input_per_1m_tokens", 0),
                output_per_1m_tokens=pricing_data.get("output_per_1m_tokens", 0)
            )
        )


@dataclass
class ProviderHealth:
    """Provider health status."""
    status: str
    latency_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class HealthStatus:
    """API health status."""
    status: str
    version: str
    providers: Dict[str, ProviderHealth] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HealthStatus:
        """Create from API response."""
        providers = {}
        for name, info in data.get("providers", {}).items():
            providers[name] = ProviderHealth(
                status=info.get("status", "unknown"),
                latency_ms=info.get("latency_ms"),
                error=info.get("error")
            )

        return cls(
            status=data.get("status", "unknown"),
            version=data.get("version", ""),
            providers=providers
        )


# ============================================================
# Config Models
# ============================================================

@dataclass
class RoutingConfig:
    """Routing configuration for requests."""
    strategy: Optional[str] = None  # cost, latency, quality
    fallback: Optional[List[str]] = None
    max_latency_ms: Optional[int] = None
    max_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.strategy:
            result["strategy"] = self.strategy
        if self.fallback:
            result["fallback"] = self.fallback
        if self.max_latency_ms:
            result["max_latency_ms"] = self.max_latency_ms
        if self.max_cost:
            result["max_cost"] = self.max_cost
        return result


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

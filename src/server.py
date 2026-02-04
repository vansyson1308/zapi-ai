"""
2api.ai - Main API Server

FastAPI-based API server for the unified AI interface.
Uses canonical error layer from src/core/errors.py
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .core.models import (
    ChatCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    Provider,
    Role,
    RoutingConfig,
    RoutingStrategy,
    Tool,
    FunctionDefinition,
    response_to_dict,
)
from .core.errors import (
    TwoApiException,
    InfraError,
    SemanticError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    InvalidRequestError,  # ✅ FIX: was InvalidRequestErr
    MissingRequiredFieldError,
    RateLimitedError,
    ProviderDownError,
    ErrorDetails,
    ErrorType,
)
from .adapters.base import AdapterConfig
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.google_adapter import GoogleAdapter
from .routing.router import Router


# ============================================================
# Pydantic Models for API (request/response validation)
# ============================================================

class MessageInput(BaseModel):
    role: str
    content: Any  # str or list of content parts
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class FunctionInput(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolInput(BaseModel):
    type: str = "function"
    function: FunctionInput


class RoutingInput(BaseModel):
    strategy: Optional[str] = None
    fallback: Optional[List[str]] = None
    max_latency_ms: Optional[int] = None
    max_cost: Optional[float] = None


class ChatCompletionInput(BaseModel):
    model: str
    messages: List[MessageInput]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[ToolInput]] = None
    tool_choice: Optional[Any] = None
    routing: Optional[RoutingInput] = None
    metadata: Optional[Dict[str, str]] = None


class EmbeddingInput(BaseModel):
    model: str
    input: Any  # str or list of str
    encoding_format: str = "float"
    dimensions: Optional[int] = None


class ImageGenerationInput(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"
    response_format: str = "url"


# ============================================================
# Global state
# ============================================================

router_instance: Optional[Router] = None


# ============================================================
# Lifespan management
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global router_instance

    adapters = {}

    # Initialize OpenAI adapter
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        adapters[Provider.OPENAI] = OpenAIAdapter(AdapterConfig(api_key=openai_key))

    # Initialize Anthropic adapter
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        adapters[Provider.ANTHROPIC] = AnthropicAdapter(AdapterConfig(api_key=anthropic_key))

    # Initialize Google adapter
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        adapters[Provider.GOOGLE] = GoogleAdapter(AdapterConfig(api_key=google_key))

    if not adapters:
        print("WARNING: No API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

    router_instance = Router(adapters)

    # Run health check
    if adapters:
        health = await router_instance.check_all_health()
        for provider, status in health.items():
            print(f"  {provider.value}: {'✓ healthy' if status.is_healthy else '✗ unhealthy'}")

    print("2api.ai server started")

    yield

    # Shutdown: Close adapters
    for adapter in adapters.values():
        await adapter.close()

    print("2api.ai server stopped")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="2api.ai",
    description="Unified AI API - Access OpenAI, Anthropic, and Google through a single interface",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request Context
# ============================================================

class RequestContext:
    """Request context with tracing information."""
    def __init__(self):
        self.request_id = f"req_{uuid.uuid4().hex[:24]}"
        self.trace_id = f"trace_{uuid.uuid4().hex[:24]}"
        self.tenant_id: Optional[str] = None
        self.api_key_id: Optional[str] = None


# ============================================================
# Authentication
# ============================================================

async def verify_api_key(
    authorization: Optional[str] = Header(None),
    request: Request | None = None
) -> tuple[str, RequestContext]:
    """
    Verify API key from Authorization header.
    Returns (api_key, request_context).
    """
    ctx = RequestContext()

    if not authorization:
        raise MissingAPIKeyError(request_id=ctx.request_id)

    # Extract bearer token
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization

    # Validate key format (in production, check against database)
    if not api_key.startswith("2api_"):
        raise InvalidAPIKeyError(
            message="Invalid API key format. Keys should start with '2api_'",
            request_id=ctx.request_id
        )

    return api_key, ctx


async def get_auth(
    authorization: Optional[str] = Header(None)
) -> RequestContext:
    """Dependency that returns request context with validated auth."""
    _, ctx = await verify_api_key(authorization=authorization)
    return ctx


def get_router() -> Router:
    """Get the router instance."""
    if router_instance is None:
        ctx = RequestContext()
        raise InfraError(
            ErrorDetails(
                code="service_unavailable",
                message="Router not initialized",
                type=ErrorType.INFRA,
                request_id=ctx.request_id,
                retryable=True,
                retry_after=5
            ),
            status_code=503
        )
    return router_instance


def add_response_headers(response: JSONResponse, ctx: RequestContext) -> JSONResponse:
    """Add standard headers to response."""
    response.headers["X-Request-Id"] = ctx.request_id
    response.headers["X-Trace-Id"] = ctx.trace_id
    return response


# ============================================================
# Helper functions
# ============================================================

def convert_message_input(msg: MessageInput) -> Message:
    """Convert API input to internal Message model."""
    role = Role(msg.role)
    return Message(
        role=role,
        content=msg.content,
        name=msg.name,
        tool_call_id=msg.tool_call_id
    )


def convert_tool_input(tool: ToolInput) -> Tool:
    """Convert API input to internal Tool model."""
    return Tool(
        type=tool.type,
        function=FunctionDefinition(
            name=tool.function.name,
            description=tool.function.description,
            parameters=tool.function.parameters
        )
    )


def convert_routing_input(routing: Optional[RoutingInput]) -> Optional[RoutingConfig]:
    """Convert API input to internal RoutingConfig."""
    if not routing:
        return None

    strategy = None
    if routing.strategy:
        strategy = RoutingStrategy(routing.strategy)

    return RoutingConfig(
        strategy=strategy,
        fallback=routing.fallback,
        max_latency_ms=routing.max_latency_ms,
        max_cost=routing.max_cost
    )


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    router = get_router()
    health = await router.check_all_health()

    all_healthy = all(h.is_healthy for h in health.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": "1.0.0",
        "providers": {
            provider.value: {
                "status": "healthy" if h.is_healthy else "unhealthy",
                "latency_ms": h.avg_latency_ms
            }
            for provider, h in health.items()
        }
    }


@app.get("/v1/models")
async def list_models(
    provider: Optional[str] = None,
    capability: Optional[str] = None,
    ctx: RequestContext = Depends(get_auth)
):
    """List available models."""
    router = get_router()
    models = router.list_all_models()

    if provider:
        models = [m for m in models if m.provider.value == provider]

    if capability:
        models = [m for m in models if m.supports(capability)]

    response = JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": m.id,
                "provider": m.provider.value,
                "name": m.name,
                "capabilities": m.capabilities,
                "context_window": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "pricing": {
                    "input_per_1m_tokens": m.pricing.input_per_1m_tokens,
                    "output_per_1m_tokens": m.pricing.output_per_1m_tokens
                }
            }
            for m in models
        ]
    })
    return add_response_headers(response, ctx)


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    body: ChatCompletionInput,
    ctx: RequestContext = Depends(get_auth)
):
    """Create a chat completion."""
    router = get_router()

    messages = [convert_message_input(m) for m in body.messages]
    tools = [convert_tool_input(t) for t in body.tools] if body.tools else None
    routing = convert_routing_input(body.routing)

    internal_request = ChatCompletionRequest(
        model=body.model,
        messages=messages,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        stream=body.stream,
        tools=tools,
        tool_choice=body.tool_choice,
        routing=routing,
        metadata=body.metadata
    )

    # Handle streaming
    if body.stream:
        async def generate():
            # Try provider-specific adapter if model is namespaced: "openai/gpt-4o" etc.
            if "/" in body.model:
                provider_name = body.model.split("/")[0]
                try:
                    provider = Provider(provider_name)
                    adapter = router.adapters.get(provider)
                    if adapter:
                        async for chunk in adapter.chat_completion_stream(internal_request):
                            yield chunk
                        return
                except (ValueError, KeyError):
                    pass

            # Fallback to OpenAI if available
            if Provider.OPENAI in router.adapters:
                adapter = router.adapters[Provider.OPENAI]
                async for chunk in adapter.chat_completion_stream(internal_request):
                    yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-Id": ctx.request_id,
                "X-Trace-Id": ctx.trace_id
            }
        )

    # Non-streaming
    try:
        response, decision = await router.route_chat(internal_request)

        json_response = JSONResponse(
            content=response_to_dict(response),
            headers={
                "X-Request-Id": ctx.request_id,
                "X-Trace-Id": ctx.trace_id,
                "X-Provider": response.provider,
                "X-Latency-Ms": str(response._2api.latency_ms) if response._2api else "0",
                "X-Cost-Usd": str(response._2api.cost_usd) if response._2api else "0",
            }
        )

        if response._2api and response._2api.routing_decision:
            if response._2api.routing_decision.fallback_used:
                json_response.headers["X-Fallback-Attempted"] = "true"

        return json_response

    except TwoApiException:
        raise
    except Exception as e:
        raise InfraError(
            ErrorDetails(
                code="internal_error",
                message=str(e),
                type=ErrorType.INFRA,
                request_id=ctx.request_id,
                retryable=True
            ),
            status_code=500
        )


@app.post("/v1/embeddings")
async def create_embedding(
    body: EmbeddingInput,
    ctx: RequestContext = Depends(get_auth)
):
    """Create embeddings."""
    router = get_router()

    internal_request = EmbeddingRequest(
        model=body.model,
        input=body.input,
        encoding_format=body.encoding_format,
        dimensions=body.dimensions
    )

    try:
        response, decision = await router.route_embedding(internal_request)
        return add_response_headers(JSONResponse(content={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": d.embedding,
                    "index": d.index
                }
                for d in response.data
            ],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }), ctx)
    except NotImplementedError as e:
        raise SemanticError(
            ErrorDetails(
                code="unsupported_operation",
                message=str(e),
                type=ErrorType.SEMANTIC,
                request_id=ctx.request_id,
                retryable=False
            ),
            status_code=400
        )
    except TwoApiException:
        raise
    except Exception as e:
        raise InfraError(
            ErrorDetails(
                code="internal_error",
                message=str(e),
                type=ErrorType.INFRA,
                request_id=ctx.request_id,
                retryable=True
            ),
            status_code=500
        )


@app.post("/v1/images/generations")
async def create_image(
    body: ImageGenerationInput,
    ctx: RequestContext = Depends(get_auth)
):
    """Generate images."""
    router = get_router()

    internal_request = ImageGenerationRequest(
        model=body.model,
        prompt=body.prompt,
        n=body.n,
        size=body.size,
        quality=body.quality,
        style=body.style,
        response_format=body.response_format
    )

    try:
        response, decision = await router.route_image(internal_request)
        return add_response_headers(JSONResponse(content={
            "created": response.created,
            "data": [
                {
                    "url": d.url,
                    "b64_json": d.b64_json,
                    "revised_prompt": d.revised_prompt
                }
                for d in response.data
            ]
        }), ctx)
    except NotImplementedError as e:
        raise SemanticError(
            ErrorDetails(
                code="unsupported_operation",
                message=str(e),
                type=ErrorType.SEMANTIC,
                request_id=ctx.request_id,
                retryable=False
            ),
            status_code=400
        )
    except TwoApiException:
        raise
    except Exception as e:
        raise InfraError(
            ErrorDetails(
                code="internal_error",
                message=str(e),
                type=ErrorType.INFRA,
                request_id=ctx.request_id,
                retryable=True
            ),
            status_code=500
        )


@app.get("/v1/usage")
async def get_usage(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: Optional[str] = None,
    ctx: RequestContext = Depends(get_auth)
):
    """Get usage statistics."""
    router = get_router()
    stats = router.get_stats()

    return add_response_headers(JSONResponse(content={
        "object": "usage",
        "data": [],
        "summary": {
            "providers": stats,
            "note": "Real usage tracking requires database integration"
        }
    }), ctx)


@app.get("/v1/stats")
async def get_stats(ctx: RequestContext = Depends(get_auth)):
    """Get routing statistics."""
    router = get_router()
    return add_response_headers(JSONResponse(content=router.get_stats()), ctx)


# ============================================================
# Error handlers
# ============================================================

@app.exception_handler(TwoApiException)
async def twoapi_exception_handler(request: Request, exc: TwoApiException):
    """Handle all 2api.ai canonical errors."""
    headers = {
        "X-Request-Id": exc.error.request_id,
        "X-Trace-Id": exc.error.request_id.replace("req_", "trace_"),
        "X-Error-Type": exc.error.type.value,
        "X-Error-Code": exc.error.code,
    }

    if exc.error.retry_after:
        headers["Retry-After"] = str(exc.error.retry_after)

    if exc.error.provider:
        headers["X-Provider"] = exc.error.provider

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.error.to_dict(),
        headers=headers
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions."""
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": str(exc.detail) if isinstance(exc.detail, str) else exc.detail.get("message", "Unknown error"),
                "type": "semantic_error" if exc.status_code < 500 else "infra_error",
                "request_id": request_id,
                "retryable": exc.status_code >= 500
            }
        },
        headers={
            "X-Request-Id": request_id,
            "X-Trace-Id": request_id.replace("req_", "trace_")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    import traceback
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "type": "infra_error",
                "request_id": request_id,
                "retryable": True
            }
        },
        headers={
            "X-Request-Id": request_id,
            "X-Trace-Id": request_id.replace("req_", "trace_")
        }
    )


# ============================================================
# Run server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

"""
2api.ai - Chat Completions API

Endpoints for chat completion requests.
Compatible with OpenAI's Chat Completions API.
"""

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...core.models import (
    ChatCompletionRequest as InternalRequest,
    Message,
    Role,
    Tool,
    FunctionDefinition,
    RoutingConfig,
    RoutingStrategy,
    response_to_dict,
)
from ...core.errors import (
    TwoApiException,
    InfraError,
    SemanticError,
    ErrorDetails,
    ErrorType,
    create_stream_error_chunk,
)
from ...auth.middleware import get_auth_context
from ...db.models import AuthContext
from ...routing.router import Router
from ...usage import (
    OperationType,
    start_tracking,
    complete_tracking,
    estimate_request_cost,
)

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    MessageInput,
    ToolDefinition,
    RoutingConfig as RoutingInput,
)
from ..dependencies import get_router, add_standard_headers, start_request_tracking, check_rate_limits


router = APIRouter(prefix="/v1", tags=["chat"])


# ============================================================
# Conversion Helpers
# ============================================================

def convert_message(msg: MessageInput) -> Message:
    """Convert API message to internal format."""
    role = Role(msg.role.value)

    # Handle tool calls
    tool_calls = None
    if msg.tool_calls:
        from ...core.models import ToolCall, FunctionCall
        tool_calls = [
            ToolCall(
                id=tc.id,
                type=tc.type,
                function=FunctionCall(
                    name=tc.function.name,
                    arguments=tc.function.arguments
                )
            )
            for tc in msg.tool_calls
        ]

    return Message(
        role=role,
        content=msg.content,
        name=msg.name,
        tool_call_id=msg.tool_call_id,
        tool_calls=tool_calls
    )


def convert_tool(tool: ToolDefinition) -> Tool:
    """Convert API tool to internal format."""
    return Tool(
        type=tool.type,
        function=FunctionDefinition(
            name=tool.function.name,
            description=tool.function.description,
            parameters=tool.function.parameters
        )
    )


def convert_routing(routing: RoutingInput) -> RoutingConfig:
    """Convert API routing to internal format."""
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
# Chat Completions Endpoint
# ============================================================

@router.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    auth: AuthContext = Depends(get_auth_context),
    _: None = Depends(check_rate_limits),
    router_instance: Router = Depends(get_router)
):
    """
    Create a chat completion.

    This endpoint is compatible with OpenAI's Chat Completions API
    with additional 2api.ai features like routing and multi-provider support.

    **Model Format:**
    - `openai/gpt-4o` - Specific provider and model
    - `anthropic/claude-3-5-sonnet` - Anthropic model
    - `google/gemini-1.5-pro` - Google model
    - `auto` - Let 2api.ai choose the best model

    **Streaming:**
    Set `stream: true` to receive Server-Sent Events (SSE).
    """
    # Convert to internal format
    messages = [convert_message(m) for m in body.messages]
    tools = [convert_tool(t) for t in body.tools] if body.tools else None
    routing = convert_routing(body.routing) if body.routing else None

    internal_request = InternalRequest(
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
        return await _handle_streaming_request(
            internal_request,
            auth,
            router_instance
        )

    # Handle non-streaming
    return await _handle_non_streaming_request(
        internal_request,
        auth,
        router_instance
    )


async def _handle_streaming_request(
    request: InternalRequest,
    auth: AuthContext,
    router_instance: Router
) -> StreamingResponse:
    """Handle streaming chat completion request via router orchestration."""

    async def generate() -> AsyncIterator[str]:
        tracker = start_request_tracking(auth, request.model, OperationType.CHAT_STREAM)

        try:
            async for chunk in router_instance.route_chat_stream(request, auth.request_id):
                if "\"content\"" in chunk and tracker._first_token_time is None:
                    tracker.record_first_token()
                yield chunk
            tracker.status = "success"

        except TwoApiException as e:
            tracker.set_error(e.error.code, e.error.message)
            yield create_stream_error_chunk(e)

        except Exception as e:
            error = InfraError(
                ErrorDetails(
                    code="stream_error",
                    message=str(e),
                    type=ErrorType.INFRA,
                    request_id=auth.request_id,
                    retryable=True
                ),
                status_code=500
            )
            tracker.set_error("stream_error", str(e))
            yield create_stream_error_chunk(error)

        finally:
            try:
                await complete_tracking(tracker)
            except Exception:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-Id": auth.request_id,
            "X-Trace-Id": auth.trace_id
        }
    )


async def _handle_non_streaming_request(
    request: InternalRequest,
    auth: AuthContext,
    router_instance: Router
) -> JSONResponse:
    """Handle non-streaming chat completion request."""
    tracker = start_request_tracking(auth, request.model, OperationType.CHAT)

    try:
        # Route the request
        response, decision = await router_instance.route_chat(request)

        # Update tracker with actual usage
        if response.usage:
            tracker.add_tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

        if decision:
            tracker.routing_strategy = decision.strategy_used
            if decision.fallback_used:
                for provider in decision.candidates_evaluated:
                    if provider != response.provider:
                        tracker.add_fallback(provider)

        # Complete tracking
        record = await complete_tracking(tracker)

        # Build response headers
        headers = add_standard_headers(
            {},
            auth,
            **{
                "X-Provider": response.provider,
                "X-Latency-Ms": str(record.latency_ms),
                "X-Cost-Usd": f"{record.cost_usd:.6f}",
            }
        )

        if decision and decision.fallback_used:
            headers["X-Fallback-Attempted"] = "true"

        return JSONResponse(
            content=response_to_dict(response),
            headers=headers
        )

    except TwoApiException:
        tracker.set_error("api_error", "API error occurred")
        await complete_tracking(tracker)
        raise

    except Exception as e:
        tracker.set_error("internal_error", str(e))
        await complete_tracking(tracker)

        raise InfraError(
            ErrorDetails(
                code="internal_error",
                message=str(e),
                type=ErrorType.INFRA,
                request_id=auth.request_id,
                retryable=True
            ),
            status_code=500
        )


def _get_adapter_for_model(model: str, router_instance: Router):
    """Get the appropriate adapter for a model."""
    from ...core.models import Provider

    if "/" in model:
        provider_name = model.split("/")[0]
        try:
            provider = Provider(provider_name)
            if provider in router_instance.adapters:
                return router_instance.adapters[provider]
        except ValueError:
            pass

    # Default to OpenAI if available
    if Provider.OPENAI in router_instance.adapters:
        return router_instance.adapters[Provider.OPENAI]

    # Try any available adapter
    if router_instance.adapters:
        return list(router_instance.adapters.values())[0]

    raise SemanticError(
        ErrorDetails(
            code="no_providers",
            message="No AI providers configured",
            type=ErrorType.SEMANTIC,
            request_id="",
            retryable=False
        ),
        status_code=503
    )

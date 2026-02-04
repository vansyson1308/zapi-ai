"""
2api.ai - Embeddings API

Endpoints for embedding requests.
Compatible with OpenAI's Embeddings API.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...core.models import (
    EmbeddingRequest as InternalRequest,
    Provider,
)
from ...core.errors import (
    TwoApiException,
    InfraError,
    SemanticError,
    ErrorDetails,
    ErrorType,
)
from ...auth.middleware import get_auth_context
from ...db.models import AuthContext
from ...routing.router import Router
from ...usage import (
    OperationType,
    start_tracking,
    complete_tracking,
)

from ..models import EmbeddingRequest, EmbeddingResponse
from ..dependencies import get_router, add_standard_headers, start_request_tracking


router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings")
async def create_embedding(
    body: EmbeddingRequest,
    auth: AuthContext = Depends(get_auth_context),
    router_instance: Router = Depends(get_router)
):
    """
    Create embeddings for the input text.

    This endpoint is compatible with OpenAI's Embeddings API.

    **Supported Models:**
    - `openai/text-embedding-3-small`
    - `openai/text-embedding-3-large`
    - `google/text-embedding-004`

    Note: Anthropic does not support embeddings.

    **Input Formats:**
    - Single string: `"Hello world"`
    - Array of strings: `["Hello", "World"]`
    """
    # Start tracking
    tracker = start_request_tracking(auth, body.model, OperationType.EMBEDDING)

    # Convert to internal format
    internal_request = InternalRequest(
        model=body.model,
        input=body.input,
        encoding_format=body.encoding_format,
        dimensions=body.dimensions
    )

    try:
        # Route the request
        response, decision = await router_instance.route_embedding(internal_request)

        # Update tracker
        if response.usage:
            tracker.add_tokens(input_tokens=response.usage.prompt_tokens)

        # Complete tracking
        record = await complete_tracking(tracker)

        # Build response
        headers = add_standard_headers(
            {},
            auth,
            **{
                "X-Provider": _get_provider_from_model(body.model),
                "X-Latency-Ms": str(record.latency_ms),
                "X-Cost-Usd": f"{record.cost_usd:.6f}",
            }
        )

        return JSONResponse(
            content={
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
            },
            headers=headers
        )

    except NotImplementedError as e:
        tracker.set_error("unsupported_operation", str(e))
        await complete_tracking(tracker)

        raise SemanticError(
            ErrorDetails(
                code="unsupported_operation",
                message=str(e),
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False
            ),
            status_code=400
        )

    except TwoApiException:
        tracker.set_error("api_error", "API error")
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


def _get_provider_from_model(model: str) -> str:
    """Extract provider from model string."""
    if "/" in model:
        return model.split("/")[0]
    return "openai"  # Default

"""
2api.ai - Image Generation API

Endpoints for image generation requests.
Compatible with OpenAI's Images API.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...core.models import (
    ImageGenerationRequest as InternalRequest,
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

from ..models import ImageGenerationRequest
from ..dependencies import get_router, add_standard_headers, start_request_tracking, check_rate_limits


router = APIRouter(prefix="/v1", tags=["images"])


@router.post("/images/generations")
async def create_image(
    body: ImageGenerationRequest,
    auth: AuthContext = Depends(get_auth_context),
    _: None = Depends(check_rate_limits),
    router_instance: Router = Depends(get_router)
):
    """
    Generate images from a text prompt.

    This endpoint is compatible with OpenAI's Images API.

    **Supported Models:**
    - `openai/dall-e-3` (default)
    - `openai/dall-e-2`

    Note: Google and Anthropic do not support image generation through this API.

    **Sizes (DALL-E 3):**
    - `1024x1024` (default, square)
    - `1024x1792` (portrait)
    - `1792x1024` (landscape)

    **Quality:**
    - `standard` (faster, cheaper)
    - `hd` (higher quality, more expensive)
    """
    # Start tracking
    tracker = start_request_tracking(auth, body.model, OperationType.IMAGE)

    # Convert to internal format
    internal_request = InternalRequest(
        model=body.model,
        prompt=body.prompt,
        n=body.n,
        size=body.size,
        quality=body.quality,
        style=body.style,
        response_format=body.response_format
    )

    try:
        # Route the request
        response, decision = await router_instance.route_image(internal_request)

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
                "created": response.created,
                "data": [
                    {
                        "url": d.url,
                        "b64_json": d.b64_json,
                        "revised_prompt": d.revised_prompt
                    }
                    for d in response.data
                ]
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
    return "openai"  # Default for image generation

"""
2api.ai - Models API

Endpoints for listing and getting model information.
Compatible with OpenAI's Models API.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from ...core.models import Provider
from ...auth.middleware import get_auth_context
from ...db.models import AuthContext
from ...routing.router import Router
from ...usage import list_models as list_pricing_models, get_model_price

from ..models import ModelInfo, ModelListResponse
from ..dependencies import get_router, add_standard_headers


router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(
    provider: Optional[str] = Query(
        None,
        description="Filter by provider (openai, anthropic, google)"
    ),
    capability: Optional[str] = Query(
        None,
        description="Filter by capability (chat, vision, tools, embedding, image)"
    ),
    auth: AuthContext = Depends(get_auth_context),
    router_instance: Router = Depends(get_router)
):
    """
    List all available models.

    Returns a list of models from all configured providers.
    Use query parameters to filter the results.

    **Filters:**
    - `provider`: Filter by provider name (openai, anthropic, google)
    - `capability`: Filter by capability (chat, vision, tools, embedding, image)

    **Example:**
    ```
    GET /v1/models?provider=openai&capability=chat
    ```
    """
    # Get models from router (includes all configured adapters)
    models = router_instance.list_all_models()

    # Apply filters
    if provider:
        models = [m for m in models if m.provider.value == provider]

    if capability:
        models = [m for m in models if m.supports(capability)]

    # Build response
    headers = add_standard_headers({}, auth)

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": "model",
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
        },
        headers=headers
    )


@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    auth: AuthContext = Depends(get_auth_context),
    router_instance: Router = Depends(get_router)
):
    """
    Get information about a specific model.

    **Model ID Format:**
    - Full ID: `openai/gpt-4o`
    - Short name: `gpt-4o` (assumes OpenAI provider)

    Returns model details including pricing and capabilities.
    """
    from ...core.errors import SemanticError, ErrorDetails, ErrorType

    # Get all models
    models = router_instance.list_all_models()

    # Find matching model
    model = None
    for m in models:
        if m.id == model_id or m.name == model_id:
            model = m
            break

    if not model:
        raise SemanticError(
            ErrorDetails(
                code="model_not_found",
                message=f"Model '{model_id}' not found",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False
            ),
            status_code=404
        )

    # Get pricing from catalog (may have more up-to-date info)
    price = get_model_price(model.id)

    headers = add_standard_headers({}, auth)

    return JSONResponse(
        content={
            "id": model.id,
            "object": "model",
            "provider": model.provider.value,
            "name": model.name,
            "capabilities": model.capabilities,
            "context_window": model.context_window,
            "max_output_tokens": model.max_output_tokens,
            "pricing": {
                "input_per_1m_tokens": price.input_per_1m if price else model.pricing.input_per_1m_tokens,
                "output_per_1m_tokens": price.output_per_1m if price else model.pricing.output_per_1m_tokens,
                "cached_input_per_1m": price.cached_input_per_1m if price else None,
                "batch_input_per_1m": price.batch_input_per_1m if price else None,
                "batch_output_per_1m": price.batch_output_per_1m if price else None,
            }
        },
        headers=headers
    )


@router.get("/models/compare")
async def compare_models(
    input_tokens: int = Query(..., ge=0, description="Number of input tokens"),
    output_tokens: int = Query(..., ge=0, description="Number of output tokens"),
    capability: str = Query("chat", description="Required capability"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Compare costs across models for a given usage.

    **Example:**
    ```
    GET /v1/models/compare?input_tokens=10000&output_tokens=5000&capability=chat
    ```

    Returns models sorted by cost from cheapest to most expensive.
    """
    from ...usage import compare_costs

    comparison = compare_costs(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        capability=capability
    )

    headers = add_standard_headers({}, auth)

    return JSONResponse(
        content={
            "object": "list",
            "query": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "capability": capability
            },
            "data": comparison
        },
        headers=headers
    )


@router.get("/providers")
async def list_providers(
    auth: AuthContext = Depends(get_auth_context),
    router_instance: Router = Depends(get_router)
):
    """
    List all configured providers and their status.

    Returns which providers are available and their health status.
    """
    # Get health status for all providers
    health = await router_instance.check_all_health()

    headers = add_standard_headers({}, auth)

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": provider.value,
                    "name": provider.value.capitalize(),
                    "status": "healthy" if h.is_healthy else "unhealthy",
                    "latency_ms": h.avg_latency_ms,
                    "models_count": len([
                        m for m in router_instance.list_all_models()
                        if m.provider == provider
                    ])
                }
                for provider, h in health.items()
            ]
        },
        headers=headers
    )

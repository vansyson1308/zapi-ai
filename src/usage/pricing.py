"""
2api.ai - Pricing Catalog

Comprehensive pricing information for all supported AI models.
Prices are per 1 million tokens unless otherwise noted.

Updated: January 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PricingTier(str, Enum):
    """Pricing tiers for models."""
    FREE = "free"           # Free tier (limited)
    STANDARD = "standard"   # Standard pricing
    PREMIUM = "premium"     # Premium/priority pricing
    BATCH = "batch"         # Batch/async pricing (usually 50% off)


@dataclass
class ModelPrice:
    """
    Pricing information for a single model.

    All prices are in USD per 1 million tokens.
    """
    model_id: str  # e.g., "openai/gpt-4o"
    provider: str
    model_name: str

    # Per million token pricing
    input_per_1m: float
    output_per_1m: float

    # Optional: cached input pricing (OpenAI/Anthropic)
    cached_input_per_1m: Optional[float] = None

    # Optional: batch pricing
    batch_input_per_1m: Optional[float] = None
    batch_output_per_1m: Optional[float] = None

    # Optional: image pricing (per image, not per token)
    image_price: Optional[float] = None
    image_hd_price: Optional[float] = None

    # Optional: embedding pricing
    embedding_per_1m: Optional[float] = None

    # Metadata
    context_window: int = 0
    max_output_tokens: int = 0
    capabilities: List[str] = field(default_factory=list)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        is_batch: bool = False
    ) -> float:
        """
        Calculate total cost for token usage.

        Args:
            input_tokens: Number of input tokens (excluding cached)
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens
            is_batch: Whether this is a batch request

        Returns:
            Total cost in USD
        """
        if is_batch and self.batch_input_per_1m and self.batch_output_per_1m:
            input_cost = (input_tokens / 1_000_000) * self.batch_input_per_1m
            output_cost = (output_tokens / 1_000_000) * self.batch_output_per_1m
        else:
            input_cost = (input_tokens / 1_000_000) * self.input_per_1m
            output_cost = (output_tokens / 1_000_000) * self.output_per_1m

        # Add cached token cost if applicable
        cached_cost = 0.0
        if cached_tokens > 0 and self.cached_input_per_1m:
            cached_cost = (cached_tokens / 1_000_000) * self.cached_input_per_1m

        return input_cost + output_cost + cached_cost

    def calculate_embedding_cost(self, tokens: int) -> float:
        """Calculate embedding cost."""
        if self.embedding_per_1m:
            return (tokens / 1_000_000) * self.embedding_per_1m
        return (tokens / 1_000_000) * self.input_per_1m

    def calculate_image_cost(self, count: int = 1, is_hd: bool = False) -> float:
        """Calculate image generation cost."""
        if is_hd and self.image_hd_price:
            return count * self.image_hd_price
        elif self.image_price:
            return count * self.image_price
        return 0.0


class PricingCatalog:
    """
    Centralized catalog of all model pricing.

    This is the single source of truth for model costs.
    """

    def __init__(self):
        self._prices: Dict[str, ModelPrice] = {}
        self._load_default_pricing()

    def _load_default_pricing(self):
        """Load default pricing for all supported models."""

        # ============================================================
        # OpenAI Models (as of January 2025)
        # ============================================================

        self._add_price(ModelPrice(
            model_id="openai/gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            input_per_1m=2.50,
            output_per_1m=10.00,
            cached_input_per_1m=1.25,
            batch_input_per_1m=1.25,
            batch_output_per_1m=5.00,
            context_window=128000,
            max_output_tokens=16384,
            capabilities=["chat", "vision", "tools", "json_mode"]
        ))

        self._add_price(ModelPrice(
            model_id="openai/gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            input_per_1m=0.15,
            output_per_1m=0.60,
            cached_input_per_1m=0.075,
            batch_input_per_1m=0.075,
            batch_output_per_1m=0.30,
            context_window=128000,
            max_output_tokens=16384,
            capabilities=["chat", "vision", "tools", "json_mode"]
        ))

        self._add_price(ModelPrice(
            model_id="openai/gpt-4-turbo",
            provider="openai",
            model_name="gpt-4-turbo",
            input_per_1m=10.00,
            output_per_1m=30.00,
            context_window=128000,
            max_output_tokens=4096,
            capabilities=["chat", "vision", "tools", "json_mode"]
        ))

        self._add_price(ModelPrice(
            model_id="openai/gpt-4",
            provider="openai",
            model_name="gpt-4",
            input_per_1m=30.00,
            output_per_1m=60.00,
            context_window=8192,
            max_output_tokens=8192,
            capabilities=["chat", "tools"]
        ))

        self._add_price(ModelPrice(
            model_id="openai/gpt-3.5-turbo",
            provider="openai",
            model_name="gpt-3.5-turbo",
            input_per_1m=0.50,
            output_per_1m=1.50,
            context_window=16385,
            max_output_tokens=4096,
            capabilities=["chat", "tools"]
        ))

        # OpenAI o1 models
        self._add_price(ModelPrice(
            model_id="openai/o1-preview",
            provider="openai",
            model_name="o1-preview",
            input_per_1m=15.00,
            output_per_1m=60.00,
            cached_input_per_1m=7.50,
            context_window=128000,
            max_output_tokens=32768,
            capabilities=["chat", "reasoning"]
        ))

        self._add_price(ModelPrice(
            model_id="openai/o1-mini",
            provider="openai",
            model_name="o1-mini",
            input_per_1m=3.00,
            output_per_1m=12.00,
            cached_input_per_1m=1.50,
            context_window=128000,
            max_output_tokens=65536,
            capabilities=["chat", "reasoning"]
        ))

        # OpenAI Embeddings
        self._add_price(ModelPrice(
            model_id="openai/text-embedding-3-small",
            provider="openai",
            model_name="text-embedding-3-small",
            input_per_1m=0.02,
            output_per_1m=0.0,
            embedding_per_1m=0.02,
            context_window=8191,
            max_output_tokens=0,
            capabilities=["embedding"]
        ))

        self._add_price(ModelPrice(
            model_id="openai/text-embedding-3-large",
            provider="openai",
            model_name="text-embedding-3-large",
            input_per_1m=0.13,
            output_per_1m=0.0,
            embedding_per_1m=0.13,
            context_window=8191,
            max_output_tokens=0,
            capabilities=["embedding"]
        ))

        # OpenAI DALL-E
        self._add_price(ModelPrice(
            model_id="openai/dall-e-3",
            provider="openai",
            model_name="dall-e-3",
            input_per_1m=0.0,
            output_per_1m=0.0,
            image_price=0.04,  # 1024x1024 standard
            image_hd_price=0.08,  # 1024x1024 HD
            context_window=4000,
            max_output_tokens=0,
            capabilities=["image"]
        ))

        # ============================================================
        # Anthropic Models (as of January 2025)
        # ============================================================

        self._add_price(ModelPrice(
            model_id="anthropic/claude-3-5-sonnet",
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            input_per_1m=3.00,
            output_per_1m=15.00,
            cached_input_per_1m=0.30,  # 90% discount
            batch_input_per_1m=1.50,
            batch_output_per_1m=7.50,
            context_window=200000,
            max_output_tokens=8192,
            capabilities=["chat", "vision", "tools", "computer_use"]
        ))

        self._add_price(ModelPrice(
            model_id="anthropic/claude-3-5-haiku",
            provider="anthropic",
            model_name="claude-3-5-haiku-20241022",
            input_per_1m=0.80,
            output_per_1m=4.00,
            cached_input_per_1m=0.08,
            batch_input_per_1m=0.40,
            batch_output_per_1m=2.00,
            context_window=200000,
            max_output_tokens=8192,
            capabilities=["chat", "vision", "tools"]
        ))

        self._add_price(ModelPrice(
            model_id="anthropic/claude-3-opus",
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            input_per_1m=15.00,
            output_per_1m=75.00,
            cached_input_per_1m=1.50,
            batch_input_per_1m=7.50,
            batch_output_per_1m=37.50,
            context_window=200000,
            max_output_tokens=4096,
            capabilities=["chat", "vision", "tools"]
        ))

        self._add_price(ModelPrice(
            model_id="anthropic/claude-3-sonnet",
            provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            input_per_1m=3.00,
            output_per_1m=15.00,
            cached_input_per_1m=0.30,
            context_window=200000,
            max_output_tokens=4096,
            capabilities=["chat", "vision", "tools"]
        ))

        self._add_price(ModelPrice(
            model_id="anthropic/claude-3-haiku",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            input_per_1m=0.25,
            output_per_1m=1.25,
            cached_input_per_1m=0.03,
            batch_input_per_1m=0.125,
            batch_output_per_1m=0.625,
            context_window=200000,
            max_output_tokens=4096,
            capabilities=["chat", "vision", "tools"]
        ))

        # ============================================================
        # Google Models (as of January 2025)
        # ============================================================

        self._add_price(ModelPrice(
            model_id="google/gemini-1.5-pro",
            provider="google",
            model_name="gemini-1.5-pro",
            input_per_1m=1.25,  # Up to 128K
            output_per_1m=5.00,
            cached_input_per_1m=0.3125,  # 75% discount
            context_window=2000000,
            max_output_tokens=8192,
            capabilities=["chat", "vision", "tools", "audio"]
        ))

        self._add_price(ModelPrice(
            model_id="google/gemini-1.5-flash",
            provider="google",
            model_name="gemini-1.5-flash",
            input_per_1m=0.075,  # Up to 128K
            output_per_1m=0.30,
            cached_input_per_1m=0.01875,
            context_window=1000000,
            max_output_tokens=8192,
            capabilities=["chat", "vision", "tools", "audio"]
        ))

        self._add_price(ModelPrice(
            model_id="google/gemini-1.5-flash-8b",
            provider="google",
            model_name="gemini-1.5-flash-8b",
            input_per_1m=0.0375,
            output_per_1m=0.15,
            cached_input_per_1m=0.01,
            context_window=1000000,
            max_output_tokens=8192,
            capabilities=["chat", "vision", "tools"]
        ))

        self._add_price(ModelPrice(
            model_id="google/gemini-2.0-flash-exp",
            provider="google",
            model_name="gemini-2.0-flash-exp",
            input_per_1m=0.0,  # Free during preview
            output_per_1m=0.0,
            context_window=1000000,
            max_output_tokens=8192,
            capabilities=["chat", "vision", "tools", "audio", "realtime"]
        ))

        self._add_price(ModelPrice(
            model_id="google/gemini-1.0-pro",
            provider="google",
            model_name="gemini-1.0-pro",
            input_per_1m=0.50,
            output_per_1m=1.50,
            context_window=32760,
            max_output_tokens=8192,
            capabilities=["chat", "tools"]
        ))

        # Google Embeddings
        self._add_price(ModelPrice(
            model_id="google/text-embedding-004",
            provider="google",
            model_name="text-embedding-004",
            input_per_1m=0.00,  # Free tier
            output_per_1m=0.0,
            embedding_per_1m=0.00,
            context_window=2048,
            max_output_tokens=0,
            capabilities=["embedding"]
        ))

        # Add aliases
        self._add_alias("gpt-4o", "openai/gpt-4o")
        self._add_alias("gpt-4o-mini", "openai/gpt-4o-mini")
        self._add_alias("gpt-4-turbo", "openai/gpt-4-turbo")
        self._add_alias("gpt-4", "openai/gpt-4")
        self._add_alias("gpt-3.5-turbo", "openai/gpt-3.5-turbo")
        self._add_alias("claude-3-5-sonnet", "anthropic/claude-3-5-sonnet")
        self._add_alias("claude-3-opus", "anthropic/claude-3-opus")
        self._add_alias("claude-3-haiku", "anthropic/claude-3-haiku")
        self._add_alias("gemini-1.5-pro", "google/gemini-1.5-pro")
        self._add_alias("gemini-1.5-flash", "google/gemini-1.5-flash")

    def _add_price(self, price: ModelPrice):
        """Add a price to the catalog."""
        self._prices[price.model_id] = price

    def _add_alias(self, alias: str, model_id: str):
        """Add an alias that points to an existing model."""
        if model_id in self._prices:
            self._prices[alias] = self._prices[model_id]

    def get_price(self, model_id: str) -> Optional[ModelPrice]:
        """
        Get pricing for a model.

        Args:
            model_id: Full model ID (e.g., "openai/gpt-4o") or short name

        Returns:
            ModelPrice if found, None otherwise
        """
        # Try exact match
        if model_id in self._prices:
            return self._prices[model_id]

        # Try without provider prefix
        for key, price in self._prices.items():
            if price.model_name == model_id:
                return price

        return None

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        is_batch: bool = False
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            model_id: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count
            cached_tokens: Cached input token count
            is_batch: Whether batch pricing applies

        Returns:
            Cost in USD, or 0.0 if model not found
        """
        price = self.get_price(model_id)
        if not price:
            return 0.0

        return price.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            is_batch=is_batch
        )

    def list_models(self, provider: Optional[str] = None) -> List[ModelPrice]:
        """
        List all models, optionally filtered by provider.

        Args:
            provider: Filter by provider (optional)

        Returns:
            List of ModelPrice objects
        """
        result = []
        seen_ids = set()

        for key, price in self._prices.items():
            # Skip aliases (same object already added)
            if price.model_id in seen_ids:
                continue

            if provider is None or price.provider == provider:
                result.append(price)
                seen_ids.add(price.model_id)

        return result

    def get_cheapest_model(
        self,
        provider: Optional[str] = None,
        capability: Optional[str] = None
    ) -> Optional[ModelPrice]:
        """
        Get the cheapest model matching criteria.

        Args:
            provider: Filter by provider
            capability: Required capability

        Returns:
            Cheapest matching model
        """
        models = self.list_models(provider)

        if capability:
            models = [m for m in models if capability in m.capabilities]

        if not models:
            return None

        # Sort by average cost (input + output)
        return min(models, key=lambda m: m.input_per_1m + m.output_per_1m)

    def compare_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        capability: str = "chat"
    ) -> List[Dict[str, Any]]:
        """
        Compare costs across all models for given usage.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            capability: Required capability

        Returns:
            List of models with costs, sorted by cost
        """
        results = []

        for price in self.list_models():
            if capability not in price.capabilities:
                continue

            cost = price.calculate_cost(input_tokens, output_tokens)
            results.append({
                "model_id": price.model_id,
                "provider": price.provider,
                "model_name": price.model_name,
                "cost_usd": cost,
                "input_per_1m": price.input_per_1m,
                "output_per_1m": price.output_per_1m
            })

        return sorted(results, key=lambda x: x["cost_usd"])


# Global catalog instance
_catalog = PricingCatalog()


def get_model_price(model_id: str) -> Optional[ModelPrice]:
    """Get pricing for a model."""
    return _catalog.get_price(model_id)


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    is_batch: bool = False
) -> float:
    """Calculate cost for a request."""
    return _catalog.calculate_cost(
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        is_batch=is_batch
    )


def list_models(provider: Optional[str] = None) -> List[ModelPrice]:
    """List all available models."""
    return _catalog.list_models(provider)


def compare_costs(
    input_tokens: int,
    output_tokens: int,
    capability: str = "chat"
) -> List[Dict[str, Any]]:
    """Compare costs across models."""
    return _catalog.compare_costs(input_tokens, output_tokens, capability)

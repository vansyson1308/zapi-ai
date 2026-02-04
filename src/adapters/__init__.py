"""
2api.ai Adapters Module

Provider-specific adapters that translate between the unified
2api.ai format and each provider's native API format.
"""

from .base import BaseAdapter, AdapterConfig, ProviderHealth
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .google_adapter import GoogleAdapter

__all__ = [
    "BaseAdapter",
    "AdapterConfig",
    "ProviderHealth",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleAdapter",
]


def get_adapter(provider: str, config: AdapterConfig) -> BaseAdapter:
    """
    Factory function to get the appropriate adapter for a provider.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google")
        config: Adapter configuration with API key
        
    Returns:
        Configured adapter instance
        
    Raises:
        ValueError: If provider is not supported
    """
    adapters = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "google": GoogleAdapter,
    }
    
    adapter_class = adapters.get(provider.lower())
    if not adapter_class:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return adapter_class(config)

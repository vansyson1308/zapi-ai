"""
2api.ai SDK - Utility Functions

Convenience functions for quick usage without creating a client.
"""

from typing import Dict, Generator, List, Optional, Union

from .client import TwoAPI
from .models import (
    Message,
    ChatResponse,
    EmbeddingResponse,
    ImageResponse,
)


# Default client instance (created lazily)
_default_client: Optional[TwoAPI] = None


def _get_default_client() -> TwoAPI:
    """Get or create the default client."""
    global _default_client
    if _default_client is None:
        _default_client = TwoAPI()
    return _default_client


def set_api_key(api_key: str) -> None:
    """
    Set the default API key.

    This allows using the convenience functions without
    explicitly creating a client.

    Args:
        api_key: Your 2api.ai API key.

    Example:
        >>> import twoapi
        >>> twoapi.set_api_key("2api_xxx")
        >>> response = twoapi.chat("Hello!")
    """
    global _default_client
    _default_client = TwoAPI(api_key=api_key)


def chat(
    message: Union[str, List[Message], List[Dict[str, str]]],
    model: str = "auto",
    system: Optional[str] = None,
    **kwargs
) -> ChatResponse:
    """
    Quick chat function using the default client.

    Args:
        message: The message to send.
        model: Model to use.
        system: Optional system message.
        **kwargs: Additional arguments passed to client.chat()

    Returns:
        ChatResponse with the completion.

    Example:
        >>> import twoapi
        >>> response = twoapi.chat("Hello!")
        >>> print(response.content)
    """
    return _get_default_client().chat(message, model=model, system=system, **kwargs)


def chat_stream(
    message: Union[str, List[Message], List[Dict[str, str]]],
    model: str = "auto",
    system: Optional[str] = None,
    **kwargs
) -> Generator[str, None, ChatResponse]:
    """
    Quick streaming chat function using the default client.

    Args:
        message: The message to send.
        model: Model to use.
        system: Optional system message.
        **kwargs: Additional arguments.

    Yields:
        Content chunks as strings.

    Example:
        >>> import twoapi
        >>> for chunk in twoapi.chat_stream("Tell me a story"):
        ...     print(chunk, end="")
    """
    yield from _get_default_client().chat_stream(message, model=model, system=system, **kwargs)


def embed(
    text: Union[str, List[str]],
    model: str = "openai/text-embedding-3-small",
    **kwargs
) -> EmbeddingResponse:
    """
    Quick embed function using the default client.

    Args:
        text: Text or list of texts to embed.
        model: Embedding model to use.
        **kwargs: Additional arguments.

    Returns:
        EmbeddingResponse with the embeddings.

    Example:
        >>> import twoapi
        >>> response = twoapi.embed("Hello world")
        >>> print(len(response.embeddings[0]))
    """
    return _get_default_client().embed(text, model=model, **kwargs)


def generate_image(
    prompt: str,
    model: str = "openai/dall-e-3",
    **kwargs
) -> ImageResponse:
    """
    Quick image generation function using the default client.

    Args:
        prompt: Text description of the image.
        model: Image generation model.
        **kwargs: Additional arguments.

    Returns:
        ImageResponse with image URLs.

    Example:
        >>> import twoapi
        >>> response = twoapi.generate_image("A cat in space")
        >>> print(response.urls[0])
    """
    return _get_default_client().generate_image(prompt, model=model, **kwargs)

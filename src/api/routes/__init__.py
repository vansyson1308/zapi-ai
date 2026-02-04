"""
2api.ai - API Routes

Route modules for different API endpoints.
"""

from .chat import router as chat_router
from .embeddings import router as embeddings_router
from .images import router as images_router
from .models import router as models_router

__all__ = [
    "chat_router",
    "embeddings_router",
    "images_router",
    "models_router",
]

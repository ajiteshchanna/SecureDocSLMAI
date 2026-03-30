"""
SecureDocAI - Embedding Layer
Singleton embedder with lazy import and clear error messages.
"""

import logging
from typing import List

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

_embedder = None


def get_embedder():
    """Return the singleton embedder, initializing it once."""
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "Embedding package missing.\n"
                "Run: pip install langchain-huggingface sentence-transformers"
            )

    print(f"  🔧 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    _embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    print("  ✓ Embedding model ready.")
    return _embedder


def embed_query(query: str) -> List[float]:
    return get_embedder().embed_query(query)
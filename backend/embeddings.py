"""
SecureDocAI - Embedding Layer
Converts text chunks into dense vector embeddings using a local SentenceTransformer model.
Fully offline — no API calls.
"""

import logging
from typing import List, Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

# ─── SINGLETON EMBEDDER ───────────────────────────────────────────────────────

_embedder: Optional[HuggingFaceEmbeddings] = None


def get_embedder(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
) -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.
    Uses singleton pattern to avoid reloading the model multiple times.
    """
    global _embedder

    if _embedder is None:
        logger.info(f"Loading embedding model: {model_name} on {device}")
        print(f"  🔧 Loading embedding model: {model_name}...")
        _embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,   # Cosine similarity ready
                "batch_size": 32,
            },
        )
        print(f"  ✓ Embedding model loaded.")

    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings into dense vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (List[float] per text).
    """
    embedder = get_embedder()
    try:
        embeddings = embedder.embed_documents(texts)
        logger.info(f"Embedded {len(texts)} texts → dim={len(embeddings[0])}")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Args:
        query: The user's natural language question.

    Returns:
        A single embedding vector.
    """
    embedder = get_embedder()
    try:
        return embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise


def get_embedding_info() -> dict:
    """Return metadata about the loaded embedding model."""
    embedder = get_embedder()
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "device": EMBEDDING_DEVICE,
        "normalize": True,
    }
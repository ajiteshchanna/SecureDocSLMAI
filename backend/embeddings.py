"""
SecureDocAI - Embedding Layer
Singleton embedder — loaded once, reused everywhere.

Offline-safe: uses local_files_only=True so it never attempts
a network call when the model is already cached.
"""

import os
import logging
from typing import List
from pathlib import Path

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

_embedder = None


def _find_local_model_path() -> str | None:
    """
    Check if the model exists in any of the standard HuggingFace cache locations.
    Returns the local path if found, None otherwise.
    """
    # Standard HuggingFace cache locations on Windows
    possible_roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "torch" / "sentence_transformers",
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME", "")) if os.environ.get("SENTENCE_TRANSFORMERS_HOME") else None,
    ]

    # Convert model name to folder pattern
    # "sentence-transformers/all-MiniLM-L6-v2" → "models--sentence-transformers--all-MiniLM-L6-v2"
    folder_name = "models--" + EMBEDDING_MODEL_NAME.replace("/", "--")

    for root in possible_roots:
        if root and root.exists():
            candidate = root / folder_name
            if candidate.exists():
                logger.info(f"Found cached model at: {candidate}")
                return str(candidate)

    return None


def get_embedder():
    """
    Return the singleton embedder.
    Tries local cache first — only attempts download if model not found locally.
    """
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

    # Check if model is cached locally
    local_path = _find_local_model_path()

    if local_path:
        # Model is cached — load fully offline, no network call
        print(f"  📦 Found cached model — loading offline...")
        try:
            _embedder = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={
                    "device": EMBEDDING_DEVICE,
                    "local_files_only": True,   # Never phone home
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                },
            )
            print("  ✅ Embedding model loaded (offline).")
            return _embedder
        except Exception as e:
            logger.warning(f"Offline load failed ({e}), trying standard load...")

    # Model not in cache or offline load failed — try standard load
    # This will download if internet is available, use cache if not
    try:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )
        print("  ✅ Embedding model ready.")
        return _embedder

    except Exception as e:
        error_msg = str(e)
        if "connect" in error_msg.lower() or "network" in error_msg.lower() \
                or "huggingface.co" in error_msg.lower():
            raise RuntimeError(
                f"\n"
                f"  ❌ Embedding model not found in local cache.\n\n"
                f"  The model needs to be downloaded ONCE while online.\n\n"
                f"  ─── FIX ──────────────────────────────────────────────\n"
                f"  1. Connect to the internet\n"
                f"  2. Run:  python download_models.py\n"
                f"  3. Wait for download to complete (~90 MB)\n"
                f"  4. Disconnect internet\n"
                f"  5. Run:  python main.py  (works offline forever now)\n"
                f"  ──────────────────────────────────────────────────────\n"
            )
        raise


def embed_query(query: str) -> List[float]:
    return get_embedder().embed_query(query)
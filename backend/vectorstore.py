"""
SecureDocAI - Vector Store Layer
Single unified FAISS index at data/vector_db/.
All langchain imports are lazy (inside functions) so startup never fails
even if packages are being installed.
"""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from config import VECTOR_DB_PATH

logger = logging.getLogger(__name__)

# In-memory cache — loaded once, reused forever
_vectorstore = None


def _index_exists() -> bool:
    return (VECTOR_DB_PATH / "index.faiss").exists() and \
           (VECTOR_DB_PATH / "index.pkl").exists()


def index_exists() -> bool:
    return _index_exists()


def build_vectorstore(chunks: list):
    """Build a fresh FAISS index from chunks and cache it in memory."""
    global _vectorstore
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        raise ImportError(
            "langchain-community not installed.\n"
            "Run: pip install langchain-community langchain-core"
        )

    from backend.embeddings import get_embedder

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    print(f"  🏗  Building FAISS index ({len(chunks)} chunks)...")
    _vectorstore = FAISS.from_documents(documents=chunks, embedding=get_embedder())
    _vectorstore.save_local(str(VECTOR_DB_PATH))
    print(f"  ✓ FAISS index saved → {VECTOR_DB_PATH}")
    return _vectorstore


def load_vectorstore():
    """Return cached vectorstore, loading from disk only on first call."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not _index_exists():
        raise FileNotFoundError(
            f"No vector index found at: {VECTOR_DB_PATH}\n"
            "Process your documents first (option 2 in the CLI)."
        )

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        raise ImportError(
            "langchain-community not installed.\n"
            "Run: pip install langchain-community langchain-core"
        )

    from backend.embeddings import get_embedder

    print("  📂 Loading FAISS index from disk...")
    _vectorstore = FAISS.load_local(
        str(VECTOR_DB_PATH),
        get_embedder(),
        allow_dangerous_deserialization=True,
    )
    print("  ✓ FAISS index loaded.")
    return _vectorstore


def refresh_vectorstore():
    """Drop in-memory cache so next call reloads from disk."""
    global _vectorstore
    _vectorstore = None
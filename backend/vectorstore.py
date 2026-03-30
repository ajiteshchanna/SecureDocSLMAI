"""
SecureDocAI - Vector Store Layer
Manages FAISS vector databases per domain (legal, sports, finance, default, custom).
All operations are local — no cloud dependency.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.embeddings import get_embedder
from config import DOMAIN_VECTOR_DB_PATHS, SUPPORTED_DOMAINS, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _get_db_path(domain: str) -> str:
    """Resolve and validate the vector DB path for a given domain."""
    domain = domain.lower().strip()
    if domain not in DOMAIN_VECTOR_DB_PATHS:
        # Allow custom paths for ad-hoc session stores
        return str(VECTOR_DB_DIR / domain)
    return DOMAIN_VECTOR_DB_PATHS[domain]


def _db_exists(domain: str) -> bool:
    """Check if a FAISS index already exists for this domain."""
    db_path = _get_db_path(domain)
    index_file = Path(db_path) / "index.faiss"
    pkl_file = Path(db_path) / "index.pkl"
    return index_file.exists() and pkl_file.exists()


# ─── VECTORSTORE OPERATIONS ───────────────────────────────────────────────────

def create_vectorstore(chunks: List[Document], domain: str) -> FAISS:
    """
    Build a new FAISS vector store from document chunks and persist it.

    Args:
        chunks: Chunked Document objects with metadata.
        domain: Target domain name (legal, sports, finance, default, custom).

    Returns:
        The created FAISS vectorstore instance.
    """
    db_path = _get_db_path(domain)
    Path(db_path).mkdir(parents=True, exist_ok=True)

    print(f"  🏗  Building FAISS index for domain: '{domain}'...")
    embedder = get_embedder()

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedder)
    vectorstore.save_local(db_path)

    print(f"  ✓ Vector store saved → {db_path}")
    logger.info(f"Created vectorstore for domain '{domain}' at {db_path} with {len(chunks)} chunks.")
    return vectorstore


def load_vectorstore(domain: str) -> FAISS:
    """
    Load an existing FAISS vector store for a domain.

    Args:
        domain: Domain name.

    Returns:
        Loaded FAISS vectorstore instance.

    Raises:
        FileNotFoundError: If no vector store exists for this domain.
    """
    db_path = _get_db_path(domain)

    if not _db_exists(domain):
        raise FileNotFoundError(
            f"No vector store found for domain '{domain}' at: {db_path}\n"
            f"Please process documents first using option 3 in the CLI."
        )

    print(f"  📂 Loading vector store: '{domain}'...")
    embedder = get_embedder()
    vectorstore = FAISS.load_local(
        db_path,
        embedder,
        allow_dangerous_deserialization=True,  # Safe: local files only
    )
    print(f"  ✓ Vector store loaded: '{domain}'")
    logger.info(f"Loaded vectorstore for domain '{domain}' from {db_path}.")
    return vectorstore


def update_vectorstore(new_chunks: List[Document], domain: str) -> FAISS:
    """
    Merge new chunks into an existing vector store.
    Creates a new one if none exists.

    Args:
        new_chunks: New document chunks to add.
        domain: Domain name.

    Returns:
        Updated FAISS vectorstore.
    """
    db_path = _get_db_path(domain)

    if _db_exists(domain):
        print(f"  🔄 Updating existing vector store: '{domain}'...")
        vectorstore = load_vectorstore(domain)
        vectorstore.add_documents(new_chunks)
        vectorstore.save_local(db_path)
        print(f"  ✓ Updated vector store with {len(new_chunks)} new chunks.")
        logger.info(f"Updated vectorstore for domain '{domain}' with {len(new_chunks)} new chunks.")
    else:
        print(f"  ℹ  No existing store found. Creating new store for domain: '{domain}'...")
        vectorstore = create_vectorstore(new_chunks, domain)

    return vectorstore


def delete_vectorstore(domain: str) -> bool:
    """
    Delete the FAISS vector store for a given domain.

    Args:
        domain: Domain name.

    Returns:
        True if deleted, False if it didn't exist.
    """
    db_path = _get_db_path(domain)

    if not _db_exists(domain):
        print(f"  ℹ  No vector store found for domain: '{domain}'")
        return False

    shutil.rmtree(db_path, ignore_errors=True)
    print(f"  🗑  Deleted vector store for domain: '{domain}'")
    logger.info(f"Deleted vectorstore for domain '{domain}' at {db_path}.")
    return True


def list_vectorstores() -> List[dict]:
    """
    List all existing vector stores with metadata.

    Returns:
        List of dicts with domain name and index size info.
    """
    results = []
    vector_db_root = Path(VECTOR_DB_DIR)

    if not vector_db_root.exists():
        return results

    for domain_dir in sorted(vector_db_root.iterdir()):
        if domain_dir.is_dir():
            faiss_file = domain_dir / "index.faiss"
            pkl_file = domain_dir / "index.pkl"
            if faiss_file.exists() and pkl_file.exists():
                size_kb = (faiss_file.stat().st_size + pkl_file.stat().st_size) / 1024
                results.append({
                    "domain": domain_dir.name,
                    "path": str(domain_dir),
                    "size_kb": round(size_kb, 1),
                })

    return results


def domain_exists(domain: str) -> bool:
    """Public helper to check if a vector store exists for a domain."""
    return _db_exists(domain)

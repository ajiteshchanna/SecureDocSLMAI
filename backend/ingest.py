"""
SecureDocAI - Document Ingestion Layer
Handles loading and chunking of PDF, DOCX, and TXT documents.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
    RAW_DOCS_DIR,
)

logger = logging.getLogger(__name__)


# ─── LOADERS ──────────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file using PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Ensure metadata is complete
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata.setdefault("page", doc.metadata.get("page", 0) + 1)
        logger.info(f"Loaded PDF: {file_path} ({len(docs)} pages)")
        return docs
    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        raise


def load_docx(file_path: str) -> List[Document]:
    """Load a DOCX file with page-aware chunking."""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["page"] = i + 1
        logger.info(f"Loaded DOCX: {file_path} ({len(docs)} sections)")
        return docs
    except Exception as e:
        logger.error(f"Failed to load DOCX {file_path}: {e}")
        raise


def load_txt(file_path: str) -> List[Document]:
    """Load a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        doc = Document(
            page_content=content,
            metadata={
                "source": os.path.basename(file_path),
                "page": 1,
            },
        )
        logger.info(f"Loaded TXT: {file_path}")
        return [doc]
    except Exception as e:
        logger.error(f"Failed to load TXT {file_path}: {e}")
        raise


# ─── DISPATCHER ───────────────────────────────────────────────────────────────

def load_document(file_path: str) -> List[Document]:
    """Dispatch loading based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def load_directory(directory: str) -> List[Document]:
    """Load all supported documents from a directory."""
    docs = []
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return docs

    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(f"No supported documents found in: {directory}")
        return docs

    for file_path in files:
        try:
            file_docs = load_document(str(file_path))
            docs.extend(file_docs)
            print(f"  ✓ Loaded: {file_path.name} ({len(file_docs)} pages/sections)")
        except Exception as e:
            print(f"  ✗ Failed: {file_path.name} → {e}")

    return docs


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks while preserving metadata.
    Uses RecursiveCharacterTextSplitter for natural boundary detection.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Preserve and enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata.setdefault("page", 1)

    logger.info(
        f"Chunked {len(documents)} document(s) → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ─── MAIN INGESTION FUNCTION ──────────────────────────────────────────────────

def ingest_documents(
    file_paths: Optional[List[str]] = None,
    directory: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Full ingestion pipeline: load → chunk → return.

    Args:
        file_paths: List of individual file paths to ingest.
        directory: Directory to scan for documents.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with metadata.
    """
    all_docs: List[Document] = []

    if file_paths:
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"  ✗ File not found: {fp}")
                continue
            try:
                docs = load_document(fp)
                all_docs.extend(docs)
                print(f"  ✓ Loaded: {os.path.basename(fp)} ({len(docs)} pages/sections)")
            except Exception as e:
                print(f"  ✗ Error loading {fp}: {e}")

    if directory:
        dir_docs = load_directory(directory)
        all_docs.extend(dir_docs)

    if not all_docs:
        raise ValueError("No documents were successfully loaded.")

    print(f"\n  📄 Total raw pages/sections: {len(all_docs)}")
    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)
    print(f"  🔪 Total chunks created: {len(chunks)}")

    return chunks


def get_document_summary(chunks: List[Document]) -> Dict[str, Any]:
    """Return a summary of ingested chunks grouped by source."""
    summary: Dict[str, Any] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in summary:
            summary[source] = {"chunks": 0, "pages": set()}
        summary[source]["chunks"] += 1
        page = chunk.metadata.get("page", 1)
        summary[source]["pages"].add(page)

    # Convert sets to sorted lists for serialization
    for src in summary:
        summary[src]["pages"] = sorted(summary[src]["pages"])
        summary[src]["page_count"] = len(summary[src]["pages"])

    return summary
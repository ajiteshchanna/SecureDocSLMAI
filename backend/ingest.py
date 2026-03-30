"""
SecureDocAI - Document Ingestion Layer
All langchain imports are lazy (inside functions) — no module-level crashes.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS, RAW_DOCS_DIR

logger = logging.getLogger(__name__)


def _make_doc(content: str, source: str, page: int):
    """Create a LangChain Document with lazy import."""
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
    return Document(page_content=content, metadata={"source": source, "page": page})


def _load_pdf(path: Path) -> list:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise ImportError("Run: pip install langchain-community pypdf")
    docs = PyPDFLoader(str(path)).load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["page"]   = doc.metadata.get("page", 0) + 1
    return docs


def _load_docx(path: Path) -> list:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        raise ImportError("Run: pip install langchain-community docx2txt")
    docs = Docx2txtLoader(str(path)).load()
    for i, doc in enumerate(docs):
        doc.metadata["source"] = path.name
        doc.metadata["page"]   = i + 1
    return docs


def _load_txt(path: Path) -> list:
    content = path.read_text(encoding="utf-8", errors="replace")
    return [_make_doc(content, path.name, 1)]


def _load_file(path: Path) -> list:
    ext = path.suffix.lower()
    if ext == ".pdf":  return _load_pdf(path)
    if ext == ".docx": return _load_docx(path)
    if ext == ".txt":  return _load_txt(path)
    raise ValueError(f"Unsupported: {ext}")


def get_raw_doc_list() -> List[Path]:
    return sorted([
        f for f in RAW_DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


def load_all_documents() -> list:
    files = get_raw_doc_list()
    if not files:
        raise FileNotFoundError(
            f"No documents found in: {RAW_DOCS_DIR}\n"
            "Upload files first using option 1 in the CLI."
        )
    all_docs = []
    for f in files:
        try:
            docs = _load_file(f)
            all_docs.extend(docs)
            print(f"  ✓ {f.name}  ({len(docs)} page/section)")
        except Exception as e:
            print(f"  ✗ {f.name}  → {e}")
    if not all_docs:
        raise ValueError("All files failed to load.")
    return all_docs


def chunk_documents(documents: list) -> list:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError("Run: pip install langchain-text-splitters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata.setdefault("source", "unknown")
        c.metadata.setdefault("page", 1)
    return chunks


def ingest_all() -> list:
    """Main entry point: scan raw_docs/ → load → chunk → return."""
    print(f"\n  📂 Scanning: {RAW_DOCS_DIR}")
    docs   = load_all_documents()
    print(f"\n  📄 Total pages/sections loaded: {len(docs)}")
    chunks = chunk_documents(docs)
    print(f"  🔪 Total chunks created: {len(chunks)}")
    return chunks


def summarize_chunks(chunks: list) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in summary:
            summary[src] = {"chunks": 0, "pages": set()}
        summary[src]["chunks"] += 1
        summary[src]["pages"].add(c.metadata.get("page", 1))
    for src in summary:
        summary[src]["pages"] = sorted(summary[src]["pages"])
    return summary
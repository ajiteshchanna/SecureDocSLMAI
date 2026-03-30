"""
SecureDocAI - Document Ingestion Layer
Supports: text PDFs, scanned PDFs (OCR), DOCX, TXT, JPG, PNG.

PDF Loading Strategy (automatic, no user input needed):
  1. Try normal text extraction (PyPDFLoader) — fast
  2. Check text quality per page
  3. If majority of pages are empty/garbled → fallback to OCR automatically

All imports are lazy — no module-level crashes if packages are missing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS, RAW_DOCS_DIR

logger = logging.getLogger(__name__)

# Image extensions supported via OCR
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALL_EXTENSIONS   = set(SUPPORTED_EXTENSIONS) | IMAGE_EXTENSIONS


# ─── DOCUMENT FACTORY ─────────────────────────────────────────────────────────

def _make_doc(content: str, source: str, page: int, method: str = "text"):
    """Create a LangChain Document with source, page, and extraction method."""
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
    return Document(
        page_content=content,
        metadata={"source": source, "page": page, "extraction": method},
    )


# ─── TEXT-BASED PDF LOADER ────────────────────────────────────────────────────

def _load_pdf_text(path: Path) -> List:
    """
    Attempt normal (non-OCR) PDF text extraction via PyPDFLoader.
    Returns list of Document objects. May return empty/short docs for scanned PDFs.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise ImportError("Run: pip install langchain-community pypdf")

    docs = PyPDFLoader(str(path)).load()
    for doc in docs:
        doc.metadata["source"]     = path.name
        doc.metadata["page"]       = doc.metadata.get("page", 0) + 1
        doc.metadata["extraction"] = "text"
    return docs


# ─── OCR-BASED PDF LOADER ─────────────────────────────────────────────────────

def _load_pdf_ocr(path: Path) -> List:
    """
    Extract text from a scanned PDF using Tesseract OCR.
    Returns list of Document objects (one per page that has text).
    """
    from backend.ocr import extract_text_with_ocr

    page_results = extract_text_with_ocr(path)   # [(page_num, text), ...]
    docs = []
    for page_num, text in page_results:
        if text.strip():
            docs.append(_make_doc(text, path.name, page_num, method="ocr"))
    return docs


# ─── SMART PDF LOADER (auto-detect text vs scanned) ──────────────────────────

def _load_pdf(path: Path) -> List:
    """
    Smart PDF loader:
      Step 1 — Extract text normally
      Step 2 — Check quality of extracted text
      Step 3 — If poor quality → automatically switch to OCR
    """
    from backend.ocr import needs_ocr, ocr_status

    print(f"  📄 Loading PDF: {path.name}")

    # Step 1: Normal extraction
    try:
        docs = _load_pdf_text(path)
    except Exception as e:
        logger.warning(f"Normal PDF extraction failed ({e}), trying OCR...")
        docs = []

    # Step 2: Check text quality
    pages_text = [d.page_content for d in docs]

    if not needs_ocr(pages_text):
        # Normal text extraction was good
        print(f"       → text-based  ({len(docs)} pages)")
        return docs

    # Step 3: OCR fallback
    print(f"       → scanned PDF detected — switching to OCR...")

    status = ocr_status()
    if not status["ocr_ready"]:
        _warn_ocr_unavailable(status, path.name)
        # Return whatever text extraction got (even if poor)
        if docs:
            logger.warning(f"Returning low-quality text for {path.name} — OCR unavailable.")
            return docs
        return []

    ocr_docs = _load_pdf_ocr(path)
    if ocr_docs:
        print(f"       → OCR complete ({len(ocr_docs)} pages extracted)")
        return ocr_docs

    # OCR ran but got nothing — fall back to original text if available
    logger.warning(f"OCR produced no text for {path.name}. Using original extraction.")
    return docs


def _warn_ocr_unavailable(status: dict, filename: str):
    """Print a clear, actionable warning when OCR is not set up."""
    print(f"\n  ⚠  Scanned PDF detected: {filename}")
    print("     OCR is not fully set up. Missing components:")
    if not status["tesseract"]:
        print("     • Tesseract binary — see TESSERACT_SETUP.md")
    if not status["pymupdf"] and not status["pdf2image"]:
        print("     • PDF→image converter: pip install pymupdf pillow")
    print("     Processing with low-quality text extraction for now.\n")


# ─── DOCX LOADER ──────────────────────────────────────────────────────────────

def _load_docx(path: Path) -> List:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        raise ImportError("Run: pip install langchain-community docx2txt")

    docs = Docx2txtLoader(str(path)).load()
    for i, doc in enumerate(docs):
        doc.metadata["source"]     = path.name
        doc.metadata["page"]       = i + 1
        doc.metadata["extraction"] = "text"
    print(f"  📝 {path.name}  → text  ({len(docs)} section)")
    return docs


# ─── TXT LOADER ───────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> List:
    content = path.read_text(encoding="utf-8", errors="replace")
    print(f"  📃 {path.name}  → text  (1 section)")
    return [_make_doc(content, path.name, 1, method="text")]


# ─── IMAGE LOADER (JPG / PNG via OCR) ────────────────────────────────────────

def _load_image(path: Path) -> List:
    """Extract text from a JPG or PNG image via OCR."""
    from backend.ocr import extract_text_from_image, ocr_status

    status = ocr_status()
    if not status["ocr_ready"]:
        _warn_ocr_unavailable(status, path.name)
        return []

    text = extract_text_from_image(path)
    if not text.strip():
        print(f"  ⚠  {path.name}  → OCR extracted no text (blank or unreadable image)")
        return []

    print(f"  🖼  {path.name}  → OCR  ({len(text)} chars)")
    return [_make_doc(text, path.name, 1, method="ocr")]


# ─── UNIFIED FILE DISPATCHER ─────────────────────────────────────────────────

def _load_file(path: Path) -> List:
    ext = path.suffix.lower()
    if ext == ".pdf":                          return _load_pdf(path)
    if ext == ".docx":                         return _load_docx(path)
    if ext == ".txt":                          return _load_txt(path)
    if ext in IMAGE_EXTENSIONS:               return _load_image(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ─── DIRECTORY SCANNER ────────────────────────────────────────────────────────

def get_raw_doc_list() -> List[Path]:
    """Return all supported files currently in raw_docs/."""
    return sorted([
        f for f in RAW_DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS
    ])


def load_all_documents() -> List:
    """
    Scan raw_docs/, detect file type for each file,
    extract text (with automatic OCR fallback for scanned PDFs).
    Returns flat list of Document objects with full metadata.
    """
    files = get_raw_doc_list()
    if not files:
        raise FileNotFoundError(
            f"No documents found in: {RAW_DOCS_DIR}\n"
            "Upload files first using option 1 in the CLI."
        )

    print(f"\n  Found {len(files)} file(s) — detecting types...\n")

    all_docs: List = []
    stats = {"text": 0, "ocr": 0, "failed": 0}

    for f in files:
        try:
            docs = _load_file(f)
            if docs:
                all_docs.extend(docs)
                # Track extraction method for summary
                methods = {d.metadata.get("extraction", "text") for d in docs}
                if "ocr" in methods:
                    stats["ocr"] += 1
                else:
                    stats["text"] += 1
            else:
                stats["failed"] += 1
                print(f"  ✗ {f.name}  → no text extracted")
        except Exception as e:
            stats["failed"] += 1
            print(f"  ✗ {f.name}  → {e}")
            logger.exception(f"Failed to load {f.name}")

    if not all_docs:
        raise ValueError(
            "No text could be extracted from any document.\n"
            "If your PDFs are scanned, ensure Tesseract is installed (see TESSERACT_SETUP.md)."
        )

    print(f"\n  ─── Extraction Summary ───")
    print(f"  Text-based files : {stats['text']}")
    print(f"  OCR files        : {stats['ocr']}")
    if stats["failed"]:
        print(f"  Failed           : {stats['failed']}")

    return all_docs


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_documents(documents: List) -> List:
    """
    Split documents into overlapping chunks.
    OCR text uses slightly larger separators to handle imperfect line breaks.
    """
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
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata.setdefault("source",     "unknown")
        c.metadata.setdefault("page",       1)
        c.metadata.setdefault("extraction", "text")

    return chunks


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def ingest_all() -> List:
    """
    Full ingestion pipeline:
      Scan raw_docs/ → load all files (auto OCR if needed) → chunk → return.

    This is the single function called by the CLI (option 2).
    No arguments needed — everything is automatic.
    """
    print(f"  📂 Scanning: {RAW_DOCS_DIR}")

    # Print OCR status so user knows what's available
    _print_ocr_status()

    docs   = load_all_documents()
    print(f"\n  📄 Total pages/sections loaded : {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"  🔪 Total chunks created        : {len(chunks)}")

    return chunks


def _print_ocr_status():
    """Print a one-line OCR readiness status at processing time."""
    try:
        from backend.ocr import ocr_status
        s = ocr_status()
        if s["ocr_ready"]:
            ver = s.get("tesseract_version", "")
            engine = "PyMuPDF" if s["pymupdf"] else "pdf2image"
            print(f"  🔬 OCR ready  (Tesseract {ver} | {engine})")
        else:
            print("  ⚠  OCR not ready — scanned PDFs will use fallback text extraction")
            print("     To enable OCR: see TESSERACT_SETUP.md")
    except Exception:
        pass   # Don't let OCR status check crash the main flow


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def summarize_chunks(chunks: List) -> Dict[str, Any]:
    """Group chunk statistics by source file."""
    summary: Dict[str, Any] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in summary:
            summary[src] = {
                "chunks":     0,
                "pages":      set(),
                "extraction": c.metadata.get("extraction", "text"),
            }
        summary[src]["chunks"] += 1
        summary[src]["pages"].add(c.metadata.get("page", 1))

    for src in summary:
        summary[src]["pages"] = sorted(summary[src]["pages"])
        summary[src]["page_count"] = len(summary[src]["pages"])

    return summary
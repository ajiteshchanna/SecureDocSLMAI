"""
SecureDocAI - OCR Module
Handles text extraction from scanned PDFs and images via Tesseract.

Strategy:
  1. Try normal text extraction (fast, no OCR)
  2. If text is empty or too sparse → fall back to OCR automatically
  3. OCR: PDF pages → images (via PyMuPDF) → Tesseract → clean text

Dependencies:
  pip install pytesseract pymupdf pillow
  + Tesseract binary installed locally (see TESSERACT_SETUP.md)

Fully offline — no cloud OCR APIs.
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────

# A page with fewer than this many characters is considered "empty" / scanned
MIN_CHARS_PER_PAGE = 50

# Ratio: if meaningful chars / total chars < this, page is likely garbled OCR artifact
MIN_MEANINGFUL_RATIO = 0.3

# DPI for rendering PDF pages to images — 200 is fast, 300 is higher quality
OCR_DPI = 200

# Tesseract language — change to "hin" for Hindi, "eng+hin" for both
TESSERACT_LANG = "eng"


# ─── AVAILABILITY CHECKS ──────────────────────────────────────────────────────

def is_tesseract_available() -> bool:
    """Check if Tesseract binary is installed and reachable."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def is_pymupdf_available() -> bool:
    """Check if PyMuPDF (fitz) is installed."""
    try:
        import fitz  # noqa
        return True
    except ImportError:
        return False


def is_pdf2image_available() -> bool:
    """Check if pdf2image + poppler is available (Windows alternative)."""
    try:
        from pdf2image import convert_from_path  # noqa
        return True
    except ImportError:
        return False


# ─── PDF TEXT QUALITY CHECK ───────────────────────────────────────────────────

def _is_text_sufficient(text: str) -> bool:
    """
    Heuristic to decide if extracted text is good enough to skip OCR.

    A page is considered 'scanned' (needs OCR) if:
      - Total text is very short (< MIN_CHARS_PER_PAGE), OR
      - The ratio of alphanumeric chars is too low (garbled extraction)
    """
    text = text.strip()
    if len(text) < MIN_CHARS_PER_PAGE:
        return False

    # Count meaningful characters (letters + digits + common punctuation)
    meaningful = sum(1 for c in text if c.isalnum() or c in " .,;:!?-\n")
    ratio = meaningful / max(len(text), 1)

    return ratio >= MIN_MEANINGFUL_RATIO


def needs_ocr(pages_text: List[str]) -> bool:
    """
    Given a list of per-page text strings from normal extraction,
    return True if OCR should be used instead.

    Logic: if MORE THAN HALF the pages are text-insufficient → OCR the whole PDF.
    """
    if not pages_text:
        return True
    weak_pages = sum(1 for t in pages_text if not _is_text_sufficient(t))
    return weak_pages > len(pages_text) / 2


# ─── TEXT CLEANING ────────────────────────────────────────────────────────────

def clean_ocr_text(text: str) -> str:
    """
    Post-process raw OCR output to improve quality:
      - Remove null bytes and control characters
      - Collapse excessive whitespace
      - Fix common OCR artifacts (multiple spaces, stray symbols)
      - Preserve paragraph structure (double newlines)
    """
    if not text:
        return ""

    # Remove null bytes and non-printable control characters (except \n \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Replace multiple spaces with single space (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3+ consecutive newlines into 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just noise (single char, or only symbols)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep empty lines (paragraph separators) and lines with real content
        if not stripped or len(stripped) >= 3 and any(c.isalpha() for c in stripped):
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    return text.strip()


# ─── PDF → IMAGES (PyMuPDF — preferred on Windows, no Poppler needed) ────────

def pdf_to_images_pymupdf(pdf_path: Path, dpi: int = OCR_DPI) -> List[object]:
    """
    Convert each PDF page to a PIL Image using PyMuPDF.
    PyMuPDF doesn't require Poppler — works out of the box on Windows.

    Returns: List of PIL.Image objects (one per page).
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io
    except ImportError:
        raise ImportError(
            "PyMuPDF or Pillow not installed.\n"
            "Run: pip install pymupdf pillow"
        )

    images = []
    zoom   = dpi / 72.0   # PDF native resolution is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix  = page.get_pixmap(matrix=matrix, alpha=False)
        img  = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()

    return images


def pdf_to_images_pdf2image(pdf_path: Path, dpi: int = OCR_DPI) -> List[object]:
    """
    Fallback: Convert PDF pages to images using pdf2image (requires Poppler).
    Use this if PyMuPDF is not available.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image not installed.\n"
            "Run: pip install pdf2image\n"
            "Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases"
        )
    return convert_from_path(str(pdf_path), dpi=dpi)


def pdf_to_images(pdf_path: Path, dpi: int = OCR_DPI) -> List[object]:
    """
    Convert PDF to images — tries PyMuPDF first, falls back to pdf2image.
    """
    if is_pymupdf_available():
        return pdf_to_images_pymupdf(pdf_path, dpi)
    elif is_pdf2image_available():
        logger.info("PyMuPDF not found — using pdf2image fallback.")
        return pdf_to_images_pdf2image(pdf_path, dpi)
    else:
        raise ImportError(
            "No PDF→image converter found.\n"
            "Run: pip install pymupdf pillow\n"
            "(or: pip install pdf2image + install Poppler)"
        )


# ─── CORE OCR FUNCTION ────────────────────────────────────────────────────────

def ocr_page_image(image) -> str:
    """
    Run Tesseract OCR on a single PIL Image.
    Returns cleaned extracted text.
    """
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract not installed.\n"
            "Run: pip install pytesseract\n"
            "Also install Tesseract binary: see TESSERACT_SETUP.md"
        )

    try:
        # --psm 3 = fully automatic page segmentation (best for documents)
        # --oem 3 = use LSTM neural net engine
        config = f"--psm 3 --oem 3 -l {TESSERACT_LANG}"
        raw    = pytesseract.image_to_string(image, config=config)
        return clean_ocr_text(raw)
    except Exception as e:
        logger.error(f"OCR failed on page: {e}")
        return ""


# ─── FULL OCR PIPELINE FOR A PDF ──────────────────────────────────────────────

def extract_text_with_ocr(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Full OCR pipeline for a scanned PDF.

    Steps:
      1. Convert each page to an image (PyMuPDF / pdf2image)
      2. Run Tesseract OCR on each image
      3. Clean extracted text
      4. Return list of (page_number, text) tuples

    Args:
        pdf_path: Path to the scanned PDF file.

    Returns:
        List of (page_number, cleaned_text) — page numbers are 1-indexed.
        Pages that produce no text are skipped.
    """
    if not is_tesseract_available():
        raise RuntimeError(
            "Tesseract OCR is not installed or not in PATH.\n"
            "See TESSERACT_SETUP.md for installation instructions."
        )

    logger.info(f"Starting OCR on: {pdf_path.name}")
    print(f"  🔬 OCR mode: converting pages to images (DPI={OCR_DPI})...")

    images = pdf_to_images(pdf_path)
    total  = len(images)
    results: List[Tuple[int, str]] = []

    for i, image in enumerate(images, 1):
        print(f"  🔤 OCR page {i}/{total}...", end="\r", flush=True)
        text = ocr_page_image(image)
        if text:   # skip blank pages
            results.append((i, text))

    print()   # newline after \r progress
    extracted = len(results)
    logger.info(f"OCR complete: {extracted}/{total} pages had text.")
    print(f"  ✓ OCR complete — {extracted}/{total} pages extracted")

    return results


# ─── OCR FOR IMAGE FILES (BONUS: JPG / PNG) ───────────────────────────────────

def extract_text_from_image(image_path: Path) -> str:
    """
    Run OCR directly on a JPG or PNG image file.
    Returns cleaned text string.
    """
    if not is_tesseract_available():
        raise RuntimeError("Tesseract not available. See TESSERACT_SETUP.md")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Run: pip install pillow")

    print(f"  🔬 OCR on image: {image_path.name}...")
    img  = Image.open(str(image_path))
    text = ocr_page_image(img)
    print(f"  ✓ Extracted {len(text)} characters from {image_path.name}")
    return text


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

def ocr_status() -> dict:
    """Return availability status of all OCR components."""
    status = {
        "tesseract":  is_tesseract_available(),
        "pymupdf":    is_pymupdf_available(),
        "pdf2image":  is_pdf2image_available(),
    }

    if status["tesseract"]:
        try:
            import pytesseract
            status["tesseract_version"] = str(pytesseract.get_tesseract_version())
        except Exception:
            status["tesseract_version"] = "unknown"

    status["ocr_ready"] = status["tesseract"] and (
        status["pymupdf"] or status["pdf2image"]
    )
    return status
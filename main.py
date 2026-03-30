"""
SecureDocAI - CLI Entry Point
Interactive command-line interface for the offline document intelligence system.

Usage:
    python main.py
    python main.py --domain legal
    python main.py --domain finance --file report.pdf
"""

import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional, List

# ─── SETUP PATH ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    BANNER,
    APP_NAME,
    SUPPORTED_DOMAINS,
    RAW_DOCS_DIR,
    SLM_BACKEND,
)

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── DISPLAY HELPERS ──────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def hr(char="─", width=60):
    print(char * width)


def header(title: str):
    hr("═")
    print(f"  {title}")
    hr("═")


def success(msg: str):
    print(f"\n  SUCCESS: {msg}")


def error(msg: str):
    print(f"\n  ERROR: {msg}")


def info(msg: str):
    print(f"\n  ℹ  {msg}")


def warn(msg: str):
    print(f"\n  ⚠  {msg}")


def prompt(msg: str) -> str:
    return input(f"\n  ▶ {msg}: ").strip()


# ─── SESSION STATE ────────────────────────────────────────────────────────────

class SessionState:
    """Holds all mutable CLI session state."""

    def __init__(self):
        self.domain: str = "default"
        self.vectorstore = None
        self.slm = None
        self.cached_docs: List = []
        self.session_db_path: Optional[str] = None  # Temp path for uploaded docs
        self.is_temp_session: bool = False

    def reset_to_domain(self, domain: str):
        self.domain = domain
        self.vectorstore = None
        self.cached_docs = []
        self.is_temp_session = False
        self.session_db_path = None


session = SessionState()


# ─── MENU HANDLERS ────────────────────────────────────────────────────────────

def menu_select_domain():
    """Step 1: Choose which domain knowledge base to use."""
    header("SELECT DOMAIN")
    print("\n  Available domains:")
    for i, d in enumerate(SUPPORTED_DOMAINS, 1):
        print(f"    {i}. {d}")

    choice = prompt("Enter domain name or number")

    # Accept either name or number
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(SUPPORTED_DOMAINS):
            domain = SUPPORTED_DOMAINS[idx]
        else:
            error("Invalid number. Keeping current domain.")
            return
    elif choice.lower() in SUPPORTED_DOMAINS:
        domain = choice.lower()
    else:
        warn(f"Unknown domain '{choice}'. Using 'default'.")
        domain = "default"

    session.reset_to_domain(domain)
    success(f"Domain set to: '{domain}'")

    # Auto-load vectorstore if it exists
    _try_load_vectorstore(domain)


def _try_load_vectorstore(domain: str):
    """Attempt to load an existing vector store for the domain."""
    from backend.vectorstore import domain_exists, load_vectorstore

    if domain_exists(domain):
        try:
            print(f"\n Found existing knowledge base for '{domain}'. Loading...")
            session.vectorstore = load_vectorstore(domain)
            success(f"Knowledge base loaded: '{domain}'")
        except Exception as e:
            error(f"Could not load vector store: {e}")
    else:
        info(f"No knowledge base found for '{domain}'. Upload and process documents first (options 2 & 3).")


def menu_upload_documents():
    """Step 2: Copy user documents into the raw_docs folder for the current domain."""
    header("UPLOAD DOCUMENTS")

    domain_docs_dir = RAW_DOCS_DIR / session.domain
    domain_docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Target folder: {domain_docs_dir}")
    print(f"  Supported types: PDF, DOCX, TXT")
    print(f"\n  Enter file paths one at a time. Press Enter with no input when done.")

    uploaded = []
    while True:
        file_path = prompt("File path (or press Enter to finish)")
        if not file_path:
            break

        file_path = file_path.strip('"').strip("'")  # Handle pasted paths with quotes
        src = Path(file_path)

        if not src.exists():
            error(f"File not found: {file_path}")
            continue

        if src.suffix.lower() not in [".pdf", ".docx", ".txt"]:
            error(f"Unsupported file type: {src.suffix}")
            continue

        dest = domain_docs_dir / src.name
        try:
            shutil.copy2(str(src), str(dest))
            uploaded.append(src.name)
            success(f"Copied: {src.name} → {dest}")
        except Exception as e:
            error(f"Failed to copy {src.name}: {e}")

    if uploaded:
        print(f"\n {len(uploaded)} file(s) ready in: {domain_docs_dir}")
        info("Now go to option 3 to process these documents.")
    else:
        info("No files uploaded.")


def menu_process_documents():
    """Step 3: Ingest and index documents into the vector store."""
    header("PROCESS DOCUMENTS")

    domain_docs_dir = RAW_DOCS_DIR / session.domain
    if not domain_docs_dir.exists() or not any(domain_docs_dir.iterdir()):
        error(f"No documents found in: {domain_docs_dir}")
        info("Upload documents first using option 2.")
        return

    from backend.ingest import ingest_documents, get_document_summary
    from backend.vectorstore import update_vectorstore

    print(f"\n Scanning: {domain_docs_dir}")
    try:
        print("\n  ─── Ingestion ───")
        chunks = ingest_documents(directory=str(domain_docs_dir))

        print("\n  ─── Indexing ───")
        session.vectorstore = update_vectorstore(chunks, session.domain)
        session.cached_docs = chunks

        # Show summary
        summary = get_document_summary(chunks)
        print("\n  ─── Summary ───")
        for src, info_data in summary.items():
            print(f"    • {src}: {info_data['chunks']} chunks | {info_data['page_count']} page(s)")

        success(f"Knowledge base ready for domain: '{session.domain}'")

    except Exception as e:
        error(f"Processing failed: {e}")
        logger.exception("Document processing error")


def menu_ask_question():
    """Step 4: Ask a natural language question against the loaded knowledge base."""
    header("ASK A QUESTION")

    if session.vectorstore is None:
        error("No knowledge base loaded.")
        info("Select a domain (option 1) and process documents (option 3) first.")
        return

    # Lazy-load SLM
    if session.slm is None:
        try:
            from backend.slm_handler import get_slm
            session.slm = get_slm()
        except Exception as e:
            error(f"SLM initialization failed: {e}")
            return

    from backend.rag_pipeline import run_rag_pipeline

    print(f"\n  Domain: '{session.domain}' | SLM: {session.slm.get_model_info().get('model', '?')}")
    print("  Type 'quit' to return to the main menu.\n")

    while True:
        hr("·")
        question = prompt("Your question")

        if question.lower() in ("quit", "exit", "q", "back"):
            break

        if not question:
            continue

        start = time.time()

        try:
            result = run_rag_pipeline(
                query=question,
                vectorstore=session.vectorstore,
                all_documents=session.cached_docs if session.cached_docs else None,
                slm=session.slm,
            )
        except Exception as e:
            error(f"Pipeline error: {e}")
            logger.exception("RAG pipeline error")
            continue

        elapsed = time.time() - start

        # ─── Display Answer ────────────────────────────────────────────────
        print("\n" + "─" * 60)
        print(f" ANSWER\n")
        print(f"  {result['answer']}")

        # ─── Display Citations ─────────────────────────────────────────────
        citations = result.get("citations", [])
        if citations:
            print(f"\n  📎 SOURCES")
            seen = set()
            for c in citations:
                key = (c["source"], c["page"])
                if key not in seen:
                    seen.add(key)
                    print(f"    • {c['source']}  (Page {c['page']})")

        # ─── Debug Stats ──────────────────────────────────────────────────
        print(
            f"\n  [Retrieved: {result.get('retrieved_count', '?')} chunks | "
            f"After re-rank: {result.get('reranked_count', '?')} | "
            f"Time: {elapsed:.1f}s]"
        )
        print("─" * 60)


# ─── SYSTEM STATUS ────────────────────────────────────────────────────────────

def menu_status():
    """Show current session and system status."""
    header("SYSTEM STATUS")

    from backend.vectorstore import list_vectorstores

    print(f"\n  Active domain:    {session.domain}")
    print(f"  Vectorstore:      {'✓ Loaded' if session.vectorstore else '✗ Not loaded'}")
    print(f"  SLM:              {'✓ Ready' if session.slm else '○ Not initialized yet'}")
    print(f"  Cached docs:      {len(session.cached_docs)} chunks")
    print(f"  SLM backend:      {SLM_BACKEND}")

    stores = list_vectorstores()
    if stores:
        print(f"\n  ─── Existing Knowledge Bases ───")
        for s in stores:
            print(f"    • {s['domain']:<12} ({s['size_kb']} KB)  →  {s['path']}")
    else:
        print("\n  No knowledge bases built yet.")


# ─── MAIN MENU ────────────────────────────────────────────────────────────────

MENU_OPTIONS = {
    "1": ("Select Domain", menu_select_domain),
    "2": ("Upload Documents", menu_upload_documents),
    "3": ("Process Documents", menu_process_documents),
    "4": ("Ask Question", menu_ask_question),
    "5": ("System Status", menu_status),
    "6": ("Exit", None),
}


def show_main_menu():
    clear()
    print(BANNER)
    print(f"  Active Domain: {session.domain}  |  "
          f"KB: {'✓ Loaded' if session.vectorstore else '✗ Not loaded'}  |  "
          f"SLM: {'✓ Ready' if session.slm else '○ Standby'}\n")
    hr()
    for key, (label, _) in MENU_OPTIONS.items():
        print(f"    {key}. {label}")
    hr()


def run_cli():
    """Main CLI event loop."""
    # Optional: pre-select domain from args
    parser = argparse.ArgumentParser(description="SecureDocAI — Offline Document Intelligence")
    parser.add_argument("--domain", type=str, default=None, help="Pre-select domain on startup")
    parser.add_argument("--file", type=str, default=None, help="Document to process on startup")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "transformers"],
        default=None,
        help="Override SLM backend",
    )
    args = parser.parse_args()

    # Apply CLI overrides
    if args.backend:
        import config
        config.SLM_BACKEND = args.backend

    if args.domain:
        session.domain = args.domain.lower()
        _try_load_vectorstore(session.domain)

    while True:
        show_main_menu()
        choice = input("  ▶ Select option: ").strip()

        if choice not in MENU_OPTIONS:
            warn("Invalid option. Please choose 1–6.")
            time.sleep(1)
            continue

        label, handler = MENU_OPTIONS[choice]

        if choice == "6":
            clear()
            print(f"\n Thank you for using {APP_NAME}. Exiting...\n")
            sys.exit(0)

        clear()
        try:
            handler()
        except KeyboardInterrupt:
            info("Interrupted. Returning to main menu...")
        except Exception as e:
            error(f"Unexpected error: {e}")
            logger.exception("Unhandled error in menu handler")

        input("\n\n  Press Enter to return to the main menu...")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cli()

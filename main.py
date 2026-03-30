"""
SecureDocAI v2.0 - CLI Entry Point
Simplified 4-option menu. No domains. Drop files → process → ask.

Usage:
    python main.py
    python main.py --backend transformers
"""

import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import BANNER, APP_NAME, RAW_DOCS_DIR, SUPPORTED_EXTENSIONS, SLM_BACKEND

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s [%(name)s]: %(message)s")

# ─── DISPLAY HELPERS ──────────────────────────────────────────────────────────

def clear(): os.system("cls" if os.name == "nt" else "clear")
def hr(c="─", w=60): print(c * w)
def header(t): hr("═"); print(f"  {t}"); hr("═")
def ok(m):   print(f"\n  ✅ {m}")
def err(m):  print(f"\n  ❌ {m}")
def info(m): print(f"\n  ℹ  {m}")
def ask(m):  return input(f"\n  ▶ {m}: ").strip()


# ─── SESSION STATE ────────────────────────────────────────────────────────────
# All heavy objects are cached here — loaded once, reused forever.

class Session:
    vectorstore = None      # FAISS index (cached after first load)
    all_chunks  = []        # All chunks (cached for BM25)
    slm         = None      # SLM handler (cached after first use)

S = Session()


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _ensure_vectorstore():
    """Load vectorstore into session cache if not already loaded."""
    if S.vectorstore is not None:
        return True
    from backend.vectorstore import load_vectorstore, index_exists
    if not index_exists():
        err("No knowledge base found. Process documents first (option 2).")
        return False
    try:
        S.vectorstore = load_vectorstore()
        return True
    except Exception as e:
        err(f"Failed to load index: {e}")
        return False


def _ensure_slm():
    """Load SLM into session cache if not already loaded."""
    if S.slm is not None:
        return True
    try:
        from backend.slm_handler import get_slm
        S.slm = get_slm()
        return True
    except Exception as e:
        err(f"SLM initialization failed: {e}")
        return False


# ─── MENU HANDLERS ────────────────────────────────────────────────────────────

def menu_upload():
    """Copy user files into data/raw_docs/."""
    header("UPLOAD DOCUMENT")
    print(f"\n  Drop files into: {RAW_DOCS_DIR}")
    print("  Supported: PDF, DOCX, TXT")
    print("  Enter file paths one at a time. Blank line when done.\n")

    uploaded = []
    while True:
        path_str = ask("File path (Enter to finish)")
        if not path_str:
            break
        src = Path(path_str.strip('"').strip("'"))
        if not src.exists():
            err(f"Not found: {src}")
            continue
        if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
            err(f"Unsupported type: {src.suffix}")
            continue
        dest = RAW_DOCS_DIR / src.name
        try:
            shutil.copy2(str(src), str(dest))
            uploaded.append(src.name)
            ok(f"Copied → {dest.name}")
        except Exception as e:
            err(f"Copy failed: {e}")

    if uploaded:
        print(f"\n  📁 {len(uploaded)} file(s) added to raw_docs/")
        info("Run option 2 to process and index them.")
    else:
        info("No files uploaded.")


def menu_process():
    """Load all docs from raw_docs/, chunk, build FAISS index."""
    header("PROCESS ALL DOCUMENTS")

    from backend.ingest import ingest_all, summarize_chunks, get_raw_doc_list
    from backend.vectorstore import build_vectorstore, refresh_vectorstore

    files = get_raw_doc_list()
    if not files:
        err(f"No documents in {RAW_DOCS_DIR}. Upload files first (option 1).")
        return

    print(f"\n  Found {len(files)} file(s):")
    for f in files:
        print(f"    • {f.name}")

    try:
        print("\n  ─── Loading & Chunking ───")
        chunks = ingest_all()

        print("\n  ─── Building Index ───")
        vs = build_vectorstore(chunks)

        # Update session cache immediately — no need to reload from disk
        S.vectorstore = vs
        S.all_chunks  = chunks

        # Show per-file summary
        summary = summarize_chunks(chunks)
        print("\n  ─── Summary ───")
        for src, data in summary.items():
            print(f"    • {src}: {data['chunks']} chunks")

        ok(f"Knowledge base ready — {len(chunks)} total chunks from {len(files)} file(s)")

    except Exception as e:
        err(f"Processing failed: {e}")
        import traceback; traceback.print_exc()


def menu_ask():
    """Ask questions against the unified knowledge base."""
    header("ASK A QUESTION")

    if not _ensure_vectorstore():
        return
    if not _ensure_slm():
        return

    from backend.rag_pipeline import run_rag

    model = S.slm.info().get("model", "?")
    chunks_count = len(S.all_chunks)
    print(f"\n  Model: {model}  |  Indexed chunks: {chunks_count}")
    print("  Type 'quit' to return to the main menu.\n")

    while True:
        hr("·")
        question = ask("Your question")

        if question.lower() in ("quit", "exit", "q", "back", ""):
            break

        t0 = time.time()
        try:
            result = run_rag(
                query=question,
                vectorstore=S.vectorstore,
                all_chunks=S.all_chunks or None,
                slm=S.slm,
            )
        except Exception as e:
            err(f"Pipeline error: {e}")
            import traceback; traceback.print_exc()
            continue

        elapsed = time.time() - t0

        # ── Answer ────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("  📋 ANSWER\n")
        # Indent each line of the answer
        for line in result["answer"].splitlines():
            print(f"  {line}")

        # ── Citations ─────────────────────────────────────────────────────
        citations = result.get("citations", [])
        if citations:
            print("\n  📎 SOURCES")
            seen = set()
            for c in citations:
                key = (c["source"], c["page"])
                if key not in seen:
                    seen.add(key)
                    print(f"    • {c['source']}  (Page {c['page']})")

        # ── Stats ─────────────────────────────────────────────────────────
        print(
            f"\n  [Retrieved: {result.get('retrieved','?')}  →  "
            f"After rerank: {result.get('final','?')}  |  "
            f"Time: {elapsed:.1f}s]"
        )
        print("─" * 60)


def menu_status():
    """Show what's loaded and what's on disk."""
    header("SYSTEM STATUS")
    from backend.vectorstore import index_exists
    from backend.ingest import get_raw_doc_list

    files = get_raw_doc_list()
    print(f"\n  Raw docs folder:  {RAW_DOCS_DIR}")
    print(f"  Files in folder:  {len(files)}")
    for f in files:
        print(f"    • {f.name}")
    print(f"\n  FAISS index:      {'✓ Exists' if index_exists() else '✗ Not built yet'}")
    print(f"  Index in memory:  {'✓ Loaded' if S.vectorstore else '○ Not loaded'}")
    print(f"  Chunks cached:    {len(S.all_chunks)}")
    print(f"  SLM loaded:       {'✓ Ready' if S.slm else '○ Standby (loads on first question)'}")
    print(f"  SLM backend:      {SLM_BACKEND}")


# ─── MAIN MENU ────────────────────────────────────────────────────────────────

MENU = {
    "1": ("Upload Document",        menu_upload),
    "2": ("Process All Documents",  menu_process),
    "3": ("Ask Question",           menu_ask),
    "4": ("System Status",          menu_status),
    "5": ("Exit",                   None),
}


def show_menu():
    clear()
    print(BANNER)
    idx = "✓" if S.vectorstore else "✗"
    slm = "✓" if S.slm else "○"
    print(f"  Index: {idx}  |  SLM: {slm}  |  Chunks: {len(S.all_chunks)}\n")
    hr()
    for k, (label, _) in MENU.items():
        print(f"    {k}. {label}")
    hr()


def run_cli():
    parser = argparse.ArgumentParser(description="SecureDocAI — Offline Document Intelligence")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default=None)
    args = parser.parse_args()

    if args.backend:
        import config
        config.SLM_BACKEND = args.backend

    while True:
        show_menu()
        choice = input("  ▶ Select option: ").strip()

        if choice not in MENU:
            print("  Invalid option."); time.sleep(0.8); continue

        label, handler = MENU[choice]
        if choice == "5":
            clear()
            print(f"\n  👋 Goodbye!\n")
            sys.exit(0)

        clear()
        try:
            handler()
        except KeyboardInterrupt:
            info("Interrupted.")
        except Exception as e:
            err(f"Unexpected error: {e}")
            import traceback; traceback.print_exc()

        input("\n\n  Press Enter to return to menu...")


if __name__ == "__main__":
    run_cli()
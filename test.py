"""
SecureDocAI v2.0 - Test Suite
Tests the unified (no-domain) architecture end-to-end.

Usage:
    python test.py
    python test.py --quick      # skip model-loading tests
    python test.py --component ingest
"""

import sys, json, time, argparse, tempfile, traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ─── COLORS ───────────────────────────────────────────────────────────────────
G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; E="\033[0m"
def ok(m):   print(f"  {G}✓{E} {m}")
def fail(m): print(f"  {R}✗{E} {m}")
def skip(m): print(f"  {Y}○{E} {m}")

RESULTS = {"passed": 0, "failed": 0, "skipped": 0}

SAMPLE = {
    "doc1.txt": "Article 21 of the Indian Constitution guarantees the right to life and personal liberty. "
                "No person shall be deprived of life except by procedure established by law. " * 8,
    "doc2.txt": "A balance sheet reports a company's assets, liabilities, and shareholders equity. "
                "EBITDA stands for Earnings Before Interest Taxes Depreciation and Amortization. " * 8,
    "doc3.txt": "The offside rule in football states that a player must not be nearer to the opponent's goal "
                "than both the ball and the second-last opponent when the ball is played. " * 8,
}


def run(name, fn):
    try:
        fn(); ok(name); RESULTS["passed"] += 1
    except AssertionError as e:
        fail(f"{name} — {e}"); RESULTS["failed"] += 1
    except Exception as e:
        fail(f"{name} — {type(e).__name__}: {e}"); RESULTS["failed"] += 1


def skipped(name, reason=""):
    skip(f"{name}" + (f" ({reason})" if reason else "")); RESULTS["skipped"] += 1


# ─── TESTS ────────────────────────────────────────────────────────────────────

def test_config():
    from config import (CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME,
                        FAISS_TOP_K, BM25_TOP_K, RERANKER_TOP_K,
                        RAW_DOCS_DIR, VECTOR_DB_PATH, SLM_BACKEND,
                        MAX_CONTEXT_CHARS)
    assert CHUNK_SIZE > 0 and CHUNK_OVERLAP > 0
    assert FAISS_TOP_K == 6 and BM25_TOP_K == 4 and RERANKER_TOP_K == 4
    assert MAX_CONTEXT_CHARS > 0
    assert RAW_DOCS_DIR.exists()
    assert VECTOR_DB_PATH.exists()
    # Confirm no domain config remains
    import config
    assert not hasattr(config, "SUPPORTED_DOMAINS"), "Domain config should be removed"
    assert not hasattr(config, "DOMAIN_VECTOR_DB_PATHS"), "Domain paths should be removed"


def test_structure():
    required = [
        "main.py", "config.py", "test.py", "requirements.txt",
        "backend/__init__.py", "backend/ingest.py", "backend/embeddings.py",
        "backend/vectorstore.py", "backend/rag_pipeline.py", "backend/slm_handler.py",
        "finetuning/train_lora.py", "finetuning/dataset/train.json",
    ]
    for p in required:
        assert (BASE_DIR / p).exists(), f"Missing: {p}"


def test_ingest(tmp: Path):
    from config import RAW_DOCS_DIR
    # Write sample files into raw_docs
    for name, content in SAMPLE.items():
        (RAW_DOCS_DIR / name).write_text(content)

    from backend.ingest import load_all_documents, chunk_documents, ingest_all, summarize_chunks
    docs   = load_all_documents()
    assert len(docs) >= 3, f"Expected ≥3 docs, got {len(docs)}"

    chunks = chunk_documents(docs)
    assert len(chunks) > 0
    for c in chunks:
        assert "source" in c.metadata
        assert "page"   in c.metadata
        assert "chunk_id" in c.metadata

    summary = summarize_chunks(chunks)
    assert len(summary) >= 3   # one entry per source file

    # Cleanup
    for name in SAMPLE:
        (RAW_DOCS_DIR / name).unlink(missing_ok=True)


def test_embedder():
    from backend.embeddings import get_embedder, embed_query
    emb = get_embedder()
    assert emb is get_embedder(), "Singleton broken — different instance returned"
    v = embed_query("What is Article 21?")
    assert isinstance(v, list) and len(v) == 384


def test_vectorstore(tmp: Path):
    from config import RAW_DOCS_DIR, VECTOR_DB_PATH
    from langchain_core.documents import Document
    from backend.vectorstore import build_vectorstore, load_vectorstore, refresh_vectorstore, index_exists
    import shutil

    docs = [Document(page_content="Article 21 protects right to life.",
                     metadata={"source": "test.pdf", "page": 1, "chunk_id": 0}),
            Document(page_content="EBITDA measures operating performance.",
                     metadata={"source": "test.pdf", "page": 2, "chunk_id": 1})]

    vs = build_vectorstore(docs)
    assert vs is not None
    assert index_exists()

    # Test cache: load_vectorstore should return same object
    refresh_vectorstore()   # drop cache
    vs2 = load_vectorstore()
    assert vs2 is not None

    # Singleton: second call returns cached
    vs3 = load_vectorstore()
    assert vs2 is vs3, "load_vectorstore should return cached instance"

    results = vs2.similarity_search("Article 21", k=1)
    assert len(results) > 0
    assert "Article" in results[0].page_content


def test_merge_dedup():
    from langchain_core.documents import Document
    from backend.rag_pipeline import _merge

    a = Document(page_content="Article 21.", metadata={"source": "a.pdf", "page": 1})
    b = Document(page_content="EBITDA ratio.", metadata={"source": "b.pdf", "page": 1})
    c = Document(page_content="Article 21.", metadata={"source": "a.pdf", "page": 1})  # dup of a

    merged = _merge([(a, 0.9), (b, 0.8)], [(c, 1.2), (b, 1.0)])
    assert len(merged) == 2   # c is dup of a, second b is dup


def test_context_build():
    from langchain_core.documents import Document
    from backend.rag_pipeline import _build_context

    ranked = [
        (Document(page_content="Article 21 protects life.", metadata={"source": "const.pdf", "page": 12}), 0.95),
        (Document(page_content="EBITDA is a profitability metric.", metadata={"source": "fin.pdf", "page": 3}), 0.80),
    ]
    ctx, cites = _build_context(ranked)
    assert "Article 21" in ctx
    assert "EBITDA" in ctx
    assert len(cites) == 2
    assert cites[0]["source"] == "const.pdf"
    assert cites[0]["page"] == 12


def test_bm25():
    from langchain_core.documents import Document
    from backend.rag_pipeline import _bm25_search

    docs = [
        Document(page_content="Article 21 guarantees right to life.",
                 metadata={"source": "a.txt", "page": 1}),
        Document(page_content="Balance sheet shows company assets.",
                 metadata={"source": "b.txt", "page": 1}),
    ]
    results = _bm25_search("Article 21 right life", docs)
    # BM25 optional — ok if empty (rank-bm25 not installed)
    assert isinstance(results, list)
    if results:
        assert "Article" in results[0][0].page_content


def test_prompt_template():
    from config import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
    prompt = RAG_PROMPT_TEMPLATE.format(
        context="Article 21 protects life.",
        question="What does Article 21 protect?"
    )
    assert "Article 21" in prompt
    assert "What does Article 21" in prompt
    assert "Not found" in prompt
    assert "document" in SYSTEM_PROMPT.lower()


def test_full_pipeline_mock():
    from langchain_core.documents import Document
    from backend.ingest import chunk_documents
    from backend.vectorstore import build_vectorstore, refresh_vectorstore
    from backend.rag_pipeline import run_rag

    docs = [
        Document(page_content=" ".join(["Article 21 guarantees right to life and liberty."] * 10),
                 metadata={"source": "const.pdf", "page": 1, "chunk_id": 0}),
        Document(page_content=" ".join(["EBITDA is an operating performance measure."] * 10),
                 metadata={"source": "finance.pdf", "page": 1, "chunk_id": 1}),
    ]
    refresh_vectorstore()
    vs = build_vectorstore(docs)

    class MockSLM:
        def generate_answer(self, question, context):
            return f"MOCK: {context[:60]}..."
        def info(self): return {"backend": "mock", "model": "mock"}

    result = run_rag("What is Article 21?", vs, all_chunks=docs, slm=MockSLM())
    assert "answer" in result
    assert "citations" in result
    assert len(result["answer"]) > 0
    assert isinstance(result["citations"], list)


def test_no_domain_references():
    """Ensure domain logic is fully removed from key files."""
    domain_keywords = ["SUPPORTED_DOMAINS", "DOMAIN_VECTOR_DB_PATHS",
                       "domain_exists", "select_domain", "menu_select_domain"]
    files_to_check  = ["config.py", "backend/vectorstore.py",
                       "backend/rag_pipeline.py", "main.py"]
    for rel in files_to_check:
        content = (BASE_DIR / rel).read_text()
        for kw in domain_keywords:
            assert kw not in content, f"Found domain reference '{kw}' in {rel}"


def test_dataset():
    data = json.loads((BASE_DIR / "finetuning/dataset/train.json").read_text())
    assert isinstance(data, list) and len(data) >= 3
    for item in data:
        assert "instruction" in item and "response" in item


# ─── RUNNER ───────────────────────────────────────────────────────────────────

GROUPS = {
    "config":    [("Config (no domains)",         test_config),
                  ("Directory structure",          test_structure),
                  ("No domain references in code", test_no_domain_references)],
    "ingest":    [("Scan & load all docs",         lambda: test_ingest(None)),
                  ("Prompt template",              test_prompt_template),
                  ("Dataset format",              test_dataset)],
    "embeddings":[("Embedding singleton",         test_embedder)],
    "vectorstore":[("Build / load / cache",       lambda: test_vectorstore(None))],
    "retrieval": [("BM25 search",                 test_bm25),
                  ("Merge & dedup",               test_merge_dedup),
                  ("Context build",               test_context_build)],
    "pipeline":  [("Full pipeline (mock SLM)",    test_full_pipeline_mock)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",     action="store_true")
    parser.add_argument("--component", choices=list(GROUPS), default=None)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  SecureDocAI v2.0 — Test Suite")
    print(f"{'═'*60}\n")

    with tempfile.TemporaryDirectory() as tmp:
        for group, tests in GROUPS.items():
            if args.component and group != args.component:
                continue
            print(f"\n  ── {group.upper()} ──")
            for name, fn in tests:
                if args.quick and group == "embeddings":
                    skipped(name, "quick mode"); continue
                run(name, fn)

    total = sum(RESULTS.values())
    print(f"\n{'═'*60}")
    print(f"  {G}{RESULTS['passed']} passed{E}  "
          f"{R}{RESULTS['failed']} failed{E}  "
          f"{Y}{RESULTS['skipped']} skipped{E}  / {total} total")
    print(f"{'═'*60}\n")
    sys.exit(1 if RESULTS["failed"] else 0)


if __name__ == "__main__":
    main()
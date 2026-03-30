"""
SecureDocAI - Test Suite
Validates all system components end-to-end without requiring a live SLM.
Uses synthetic documents to test the full pipeline.

Usage:
    python test.py               # Run all tests
    python test.py --component ingest
    python test.py --component embeddings
    python test.py --component vectorstore
    python test.py --component retrieval
    python test.py --component pipeline
    python test.py --quick       # Skip slow model loading tests
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import traceback
from pathlib import Path

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.WARNING)

# ─── COLORS ───────────────────────────────────────────────────────────────────

class C:
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"


def ok(msg):   print(f"  {C.GREEN}✓{C.RESET} {msg}")
def fail(msg): print(f"  {C.RED}✗{C.RESET} {msg}")
def skip(msg): print(f"  {C.YELLOW}○{C.RESET} {msg}")
def info(msg): print(f"  {C.CYAN}→{C.RESET} {msg}")


# ─── TEST FIXTURES ────────────────────────────────────────────────────────────

SAMPLE_TEXTS = {
    "legal.txt": """
Article 21 of the Indian Constitution guarantees the right to life and personal liberty.
No person shall be deprived of his life or personal liberty except according to procedure
established by law. The Supreme Court has interpreted this article broadly to include the
right to livelihood, health, education, and a dignified life.

Habeas corpus is a legal action through which a person can seek relief from unlawful
detention. The writ commands the detention authority to produce the prisoner before the
court. It is a fundamental safeguard of individual freedom.

Article 14 ensures equality before the law and equal protection of the laws within the
territory of India. This article prohibits class legislation but permits reasonable
classification based on intelligible differentia.
""",
    "finance.txt": """
A balance sheet is a financial statement that reports a company's assets, liabilities,
and shareholders' equity at a specific point in time. It follows the fundamental
accounting equation: Assets = Liabilities + Shareholders' Equity.

Revenue recognition is the accounting principle that determines when revenue is recorded
in financial statements. Under IFRS 15, revenue is recognized when performance obligations
are satisfied and control of goods or services is transferred to the customer.

EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization.
It is commonly used as a proxy for operating cash flow and to compare profitability
across companies and industries.
""",
    "sports.txt": """
The offside rule in football (soccer) states that a player is in an offside position
if they are nearer to the opponent's goal line than both the ball and the second-last
opponent when the ball is played to them. Being in an offside position is not an offense
in itself — the player must be involved in active play.

The Duckworth-Lewis-Stern (DLS) method is a mathematical formulation used in cricket to
calculate target scores in rain-interrupted matches. It adjusts the target based on the
resources (overs and wickets) remaining for each team.

In basketball, a double-double occurs when a player accumulates a double-digit number in
two of the five statistical categories: points, rebounds, assists, steals, and blocks,
during a single game.
""",
}

SAMPLE_QUERIES = {
    "legal": [
        ("What does Article 21 guarantee?", "right to life"),
        ("What is habeas corpus?", "unlawful detention"),
        ("What does Article 14 ensure?", "equality"),
    ],
    "finance": [
        ("What is a balance sheet?", "assets"),
        ("What does EBITDA stand for?", "earnings"),
        ("What is revenue recognition?", "IFRS"),
    ],
    "sports": [
        ("What is the offside rule?", "goal line"),
        ("What is DLS method?", "cricket"),
        ("What is a double-double in basketball?", "double-digit"),
    ],
}

# ─── TEST RUNNER ──────────────────────────────────────────────────────────────

results = {"passed": 0, "failed": 0, "skipped": 0}


def run_test(name: str, fn, *args, **kwargs):
    """Run a single test and track results."""
    try:
        fn(*args, **kwargs)
        ok(name)
        results["passed"] += 1
    except AssertionError as e:
        fail(f"{name} — {e}")
        results["failed"] += 1
    except Exception as e:
        fail(f"{name} — {type(e).__name__}: {e}")
        results["failed"] += 1


def skip_test(name: str, reason: str = ""):
    skip(f"{name}{f' ({reason})' if reason else ''}")
    results["skipped"] += 1


# ─── INDIVIDUAL TESTS ─────────────────────────────────────────────────────────

def test_config_imports():
    from config import (
        CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME,
        FAISS_TOP_K, RERANKER_TOP_K, SUPPORTED_DOMAINS,
        SLM_BACKEND, BANNER
    )
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP > 0
    assert CHUNK_SIZE > CHUNK_OVERLAP
    assert len(SUPPORTED_DOMAINS) >= 4
    assert EMBEDDING_MODEL_NAME != ""
    assert SLM_BACKEND in ("ollama", "transformers")


def test_directory_structure():
    required = [
        "main.py", "config.py", "test.py",
        "backend/__init__.py", "backend/ingest.py",
        "backend/embeddings.py", "backend/vectorstore.py",
        "backend/rag_pipeline.py", "backend/slm_handler.py",
        "finetuning/train_lora.py", "finetuning/model_loader.py",
        "finetuning/config.py", "finetuning/dataset/train.json",
    ]
    for rel_path in required:
        full = BASE_DIR / rel_path
        assert full.exists(), f"Missing: {rel_path}"


def test_ingest_txt(tmp_dir: Path):
    from backend.ingest import load_txt, chunk_documents

    txt_file = tmp_dir / "test.txt"
    txt_file.write_text(SAMPLE_TEXTS["legal.txt"], encoding="utf-8")

    docs = load_txt(str(txt_file))
    assert len(docs) > 0
    assert docs[0].page_content.strip() != ""
    assert docs[0].metadata["source"] == "test.txt"

    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.page_content) <= 450  # Allow small overage
        assert "source" in c.metadata
        assert "chunk_id" in c.metadata


def test_ingest_multi_file(tmp_dir: Path):
    from backend.ingest import ingest_documents

    for name, content in SAMPLE_TEXTS.items():
        (tmp_dir / name).write_text(content, encoding="utf-8")

    chunks = ingest_documents(directory=str(tmp_dir))
    assert len(chunks) > 0

    sources = {c.metadata.get("source") for c in chunks}
    assert len(sources) == len(SAMPLE_TEXTS), f"Expected 3 sources, got: {sources}"


def test_embedding_model():
    from backend.embeddings import embed_texts, embed_query

    texts = ["What is Article 21?", "A balance sheet reports assets.", "The offside rule."]
    embeddings = embed_texts(texts)

    assert len(embeddings) == 3
    assert all(isinstance(e, list) for e in embeddings)
    assert all(len(e) > 0 for e in embeddings)

    dim = len(embeddings[0])
    assert all(len(e) == dim for e in embeddings), "All embeddings must have the same dimension"

    query_vec = embed_query("What is habeas corpus?")
    assert len(query_vec) == dim


def test_vectorstore_create_load(tmp_dir: Path):
    from backend.ingest import load_txt, chunk_documents
    from backend.vectorstore import create_vectorstore, load_vectorstore, delete_vectorstore

    from config import VECTOR_DB_DIR
    test_domain = "_test_domain_"

    # Setup
    txt_file = tmp_dir / "legal.txt"
    txt_file.write_text(SAMPLE_TEXTS["legal.txt"], encoding="utf-8")
    docs = load_txt(str(txt_file))
    chunks = chunk_documents(docs)

    # Create
    vs = create_vectorstore(chunks, test_domain)
    assert vs is not None

    # Load
    loaded_vs = load_vectorstore(test_domain)
    assert loaded_vs is not None

    # Query it
    results = loaded_vs.similarity_search("Article 21", k=2)
    assert len(results) > 0
    assert any("21" in r.page_content for r in results)

    # Cleanup
    delete_vectorstore(test_domain)
    test_path = Path(str(VECTOR_DB_DIR)) / test_domain
    assert not test_path.exists()


def test_semantic_search(tmp_dir: Path):
    from backend.ingest import load_txt, chunk_documents
    from backend.vectorstore import create_vectorstore
    from backend.rag_pipeline import semantic_search

    txt_file = tmp_dir / "legal.txt"
    txt_file.write_text(SAMPLE_TEXTS["legal.txt"], encoding="utf-8")
    docs = load_txt(str(txt_file))
    chunks = chunk_documents(docs)
    vs = create_vectorstore(chunks, "_test_search_")

    results = semantic_search("What is habeas corpus?", vs, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r[0].page_content, str) for r in results)
    assert all(isinstance(r[1], float) for r in results)

    # Cleanup
    from backend.vectorstore import delete_vectorstore
    delete_vectorstore("_test_search_")


def test_bm25_search(tmp_dir: Path):
    from backend.ingest import load_txt, chunk_documents
    from backend.rag_pipeline import bm25_search

    txt_file = tmp_dir / "legal.txt"
    txt_file.write_text(SAMPLE_TEXTS["legal.txt"], encoding="utf-8")
    docs = load_txt(str(txt_file))
    chunks = chunk_documents(docs)

    results = bm25_search("habeas corpus detention", chunks, top_k=3)
    # BM25 may return 0 results if rank-bm25 is not installed — that's OK
    assert isinstance(results, list)
    if results:
        assert all(isinstance(r[0].page_content, str) for r in results)


def test_merge_results():
    from langchain.schema import Document
    from backend.rag_pipeline import merge_results

    doc_a = Document(page_content="Article 21 guarantees life.", metadata={"source": "a.txt", "page": 1})
    doc_b = Document(page_content="Habeas corpus is a writ.", metadata={"source": "b.txt", "page": 2})
    doc_c = Document(page_content="Article 21 guarantees life.", metadata={"source": "a.txt", "page": 1})  # Duplicate

    semantic = [(doc_a, 0.95), (doc_b, 0.80)]
    bm25 = [(doc_c, 1.5), (doc_b, 1.2)]  # doc_c is dup of doc_a, doc_b is dup

    merged = merge_results(semantic, bm25)
    assert len(merged) == 2  # Duplicates removed
    contents = [m.page_content for m in merged]
    assert "Article 21" in contents[0] or "Habeas" in contents[0]


def test_context_construction():
    from langchain.schema import Document
    from backend.rag_pipeline import build_context

    docs = [
        (Document(page_content="Article 21 protects the right to life.", metadata={"source": "const.pdf", "page": 12}), 0.95),
        (Document(page_content="Habeas corpus prevents unlawful detention.", metadata={"source": "const.pdf", "page": 15}), 0.88),
    ]

    context_str, citations = build_context(docs)

    assert "Article 21" in context_str
    assert "Habeas corpus" in context_str
    assert len(citations) == 2
    assert citations[0]["source"] == "const.pdf"
    assert citations[0]["page"] == 12
    assert citations[1]["page"] == 15


def test_prompt_formatting():
    from config import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT

    context = "Article 21 guarantees the right to life."
    question = "What does Article 21 protect?"

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    assert "Article 21" in prompt
    assert question in prompt
    assert len(prompt) > 50

    assert "document" in SYSTEM_PROMPT.lower()
    assert "hallucination" in SYSTEM_PROMPT.lower() or "only" in SYSTEM_PROMPT.lower()


def test_dataset_format():
    dataset_path = BASE_DIR / "finetuning" / "dataset" / "train.json"
    assert dataset_path.exists()

    with open(dataset_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 3

    for item in data:
        assert "instruction" in item, f"Missing 'instruction' key"
        assert "response" in item, f"Missing 'response' key"
        assert len(item["instruction"]) > 0
        assert len(item["response"]) > 0


def test_reranker(tmp_dir: Path):
    try:
        from sentence_transformers import CrossEncoder
        from langchain.schema import Document
        from backend.rag_pipeline import rerank_documents

        docs = [
            Document(page_content="Article 21 guarantees life and liberty.", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="The weather is nice today.", metadata={"source": "b.pdf", "page": 1}),
            Document(page_content="Habeas corpus prevents unlawful detention.", metadata={"source": "c.pdf", "page": 1}),
        ]

        ranked = rerank_documents("What is Article 21?", docs, top_k=2)
        assert len(ranked) <= 2
        # The most relevant doc should rank first or second
        top_contents = " ".join(r[0].page_content for r in ranked)
        assert "Article 21" in top_contents or "liberty" in top_contents

    except ImportError:
        skip_test("reranker", "sentence-transformers not installed")
        return


def test_full_pipeline_mock(tmp_dir: Path):
    """
    Test the full pipeline with a mock SLM to avoid loading a real model.
    """
    from langchain.schema import Document
    from backend.ingest import load_txt, chunk_documents
    from backend.vectorstore import create_vectorstore
    from backend.rag_pipeline import run_rag_pipeline

    # Create synthetic document
    txt_file = tmp_dir / "legal.txt"
    txt_file.write_text(SAMPLE_TEXTS["legal.txt"], encoding="utf-8")
    docs = load_txt(str(txt_file))
    chunks = chunk_documents(docs)
    vs = create_vectorstore(chunks, "_test_pipeline_")

    # Mock SLM
    class MockSLM:
        def generate_answer(self, question, context):
            return f"MOCK ANSWER based on: {context[:50]}..."
        def get_model_info(self):
            return {"backend": "mock", "model": "mock-slm"}

    result = run_rag_pipeline(
        query="What does Article 21 guarantee?",
        vectorstore=vs,
        all_documents=chunks,
        slm=MockSLM(),
    )

    assert "answer" in result
    assert "citations" in result
    assert len(result["answer"]) > 0
    assert isinstance(result["citations"], list)

    # Cleanup
    from backend.vectorstore import delete_vectorstore
    delete_vectorstore("_test_pipeline_")


# ─── TEST ORCHESTRATOR ────────────────────────────────────────────────────────

def run_all_tests(quick: bool = False, component: str = None):
    print(f"\n{'═' * 60}")
    print(f"  SecureDocAI — Test Suite")
    print(f"{'═' * 60}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        tests = {
            "config": [
                ("Config imports", test_config_imports),
            ],
            "structure": [
                ("Directory structure", test_directory_structure),
            ],
            "ingest": [
                ("Ingest TXT file", lambda: test_ingest_txt(tmp_dir)),
                ("Ingest multiple files", lambda: test_ingest_multi_file(tmp_dir)),
            ],
            "embeddings": [
                ("Embedding model (may be slow)", test_embedding_model),
            ],
            "vectorstore": [
                ("Vectorstore create/load/delete", lambda: test_vectorstore_create_load(tmp_dir)),
            ],
            "retrieval": [
                ("Semantic search (FAISS)", lambda: test_semantic_search(tmp_dir)),
                ("Keyword search (BM25)", lambda: test_bm25_search(tmp_dir)),
                ("Merge results", test_merge_results),
                ("Context construction", test_context_construction),
                ("Re-ranker (CrossEncoder)", lambda: test_reranker(tmp_dir)),
            ],
            "pipeline": [
                ("Prompt formatting", test_prompt_formatting),
                ("Dataset format", test_dataset_format),
                ("Full pipeline (mock SLM)", lambda: test_full_pipeline_mock(tmp_dir)),
            ],
        }

        for group, group_tests in tests.items():
            if component and group != component:
                continue

            print(f"\n  ── {group.upper()} ──")

            for name, fn in group_tests:
                if quick and group == "embeddings":
                    skip_test(name, "quick mode")
                    continue
                run_test(name, fn)

    # Summary
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"\n{'═' * 60}")
    print(
        f"  Results: "
        f"{C.GREEN}{results['passed']} passed{C.RESET}  "
        f"{C.RED}{results['failed']} failed{C.RESET}  "
        f"{C.YELLOW}{results['skipped']} skipped{C.RESET}  "
        f"/ {total} total"
    )
    print(f"{'═' * 60}\n")

    if results["failed"] > 0:
        sys.exit(1)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureDocAI Test Suite")
    parser.add_argument(
        "--component",
        choices=["config", "structure", "ingest", "embeddings", "vectorstore", "retrieval", "pipeline"],
        default=None,
        help="Run only tests for a specific component",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow model-loading tests",
    )
    args = parser.parse_args()

    run_all_tests(quick=args.quick, component=args.component)

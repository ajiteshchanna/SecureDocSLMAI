"""
SecureDocAI - RAG Pipeline
All imports lazy — no module-level langchain dependency.
Flow: Query → FAISS(6) + BM25(4) → Merge → Rerank(4) → Context → SLM → Answer
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

from config import (
    FAISS_TOP_K, BM25_TOP_K, RERANKER_TOP_K,
    RERANKER_MODEL_NAME, MAX_CONTEXT_CHARS,
)

logger = logging.getLogger(__name__)

# ─── SINGLETON RERANKER ───────────────────────────────────────────────────────

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            import transformers
            # Suppress LOAD REPORT from unexpected keys
            transformers.logging.set_verbosity_error()
            
            print(f"  🔧 Loading re-ranker: {RERANKER_MODEL_NAME}...")
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
            print("  ✓ Re-ranker ready.")
        except Exception as e:
            logger.warning(f"Re-ranker unavailable ({e}) — will skip re-ranking.")
    return _reranker


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def _semantic_search(query: str, vs) -> list:
    try:
        return vs.similarity_search_with_score(query, k=FAISS_TOP_K)
    except Exception as e:
        logger.error(f"FAISS search error: {e}")
        return []


def _bm25_search(query: str, docs: list) -> list:
    if not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi
        corpus = [d.page_content.lower().split() for d in docs]
        bm25   = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [(d, s) for d, s in ranked[:BM25_TOP_K] if s > 0]
    except ImportError:
        logger.warning("rank-bm25 not installed — skipping keyword search.")
        return []
    except Exception as e:
        logger.error(f"BM25 error: {e}")
        return []


def _merge(sem: list, kw: list) -> list:
    seen, merged = set(), []
    for doc, _ in sem + kw:
        h = hash(doc.page_content.strip())
        if h not in seen:
            seen.add(h)
            merged.append(doc)
    return merged


def _rerank(query: str, docs: list) -> list:
    if not docs:
        return []
    reranker = get_reranker()
    if reranker is None:
        return [(d, 1.0) for d in docs[:RERANKER_TOP_K]]
    try:
        scores = reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[:RERANKER_TOP_K]
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        return [(d, 1.0) for d in docs[:RERANKER_TOP_K]]


def _build_context(ranked: list) -> Tuple[str, list]:
    parts, citations, total = [], [], 0
    for i, (doc, score) in enumerate(ranked, 1):
        text   = doc.page_content.strip()
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 100:
            break
        if len(text) > remaining:
            text = text[:remaining] + "..."
        parts.append(f"[{i}] {source} | Page {page}\n{text}")
        citations.append({"chunk": i, "source": source, "page": page,
                          "score": round(float(score), 4)})
        total += len(text)
    return "\n\n---\n\n".join(parts), citations


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_rag(query: str, vectorstore, all_chunks=None, slm=None) -> Dict[str, Any]:
    """
    Full RAG pipeline. All singletons — no reload on each call.
    """
    from backend.slm_handler import get_slm
    if slm is None:
        slm = get_slm()

    print("\n  🔍 [1/4] Semantic search...")
    sem = _semantic_search(query, vectorstore)

    print("  🔍 [2/4] Keyword search (BM25)...")
    kw  = _bm25_search(query, all_chunks or [])

    print("  🔀 [3/4] Merging results...")
    merged = _merge(sem, kw)

    if not merged:
        return {"answer": "Not found in provided documents.",
                "citations": [], "retrieved": 0, "final": 0}

    print("  📊 [4/4] Re-ranking...")
    ranked = _rerank(query, merged)

    context, citations = _build_context(ranked)

    print("  🤖 Generating answer...")
    answer = slm.generate_answer(question=query, context=context)

    return {"answer": answer, "citations": citations,
            "retrieved": len(merged), "final": len(ranked)}
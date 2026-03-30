"""
SecureDocAI - RAG Pipeline
Implements the full Retrieval-Augmented Generation pipeline:

  User Query
    → Semantic Search (FAISS)
    → Keyword Search (BM25)
    → Merge & Deduplicate
    → Re-rank (CrossEncoder)
    → Context Construction
    → SLM Generation
    → Answer + Citations
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import (
    FAISS_TOP_K,
    BM25_TOP_K,
    RERANKER_TOP_K,
    RERANKER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


# ─── SEMANTIC SEARCH ──────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    vectorstore: FAISS,
    top_k: int = FAISS_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-K semantically similar chunks from FAISS.

    Returns:
        List of (Document, score) tuples, sorted by descending similarity.
    """
    try:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        logger.info(f"Semantic search returned {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []


# ─── KEYWORD SEARCH (BM25) ────────────────────────────────────────────────────

def bm25_search(
    query: str,
    documents: List[Document],
    top_k: int = BM25_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-K keyword-matching chunks using BM25.

    Args:
        query: User question.
        documents: All documents in the current domain.
        top_k: Number of results to return.

    Returns:
        List of (Document, bm25_score) tuples.
    """
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [
            doc.page_content.lower().split() for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Pair documents with scores and sort
        scored = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [(doc, score) for doc, score in scored[:top_k] if score > 0]
        logger.info(f"BM25 search returned {len(results)} results.")
        return results

    except ImportError:
        logger.warning("rank-bm25 not installed. Skipping keyword search.")
        return []
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return []


# ─── MERGE & DEDUPLICATE ──────────────────────────────────────────────────────

def merge_results(
    semantic_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
) -> List[Document]:
    """
    Merge semantic and BM25 results, deduplicate by content hash.

    Returns:
        Deduplicated list of unique Documents.
    """
    seen_hashes = set()
    merged = []

    # Prioritize semantic results first, then BM25
    all_results = semantic_results + bm25_results

    for doc, _ in all_results:
        content_hash = hash(doc.page_content.strip())
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            merged.append(doc)

    logger.info(
        f"Merged {len(semantic_results)} semantic + {len(bm25_results)} BM25 "
        f"→ {len(merged)} unique chunks."
    )
    return merged


# ─── CROSS-ENCODER RE-RANKER ──────────────────────────────────────────────────

_reranker = None

def get_reranker():
    """Lazy-load the CrossEncoder re-ranker (singleton)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"  🔧 Loading re-ranker: {RERANKER_MODEL_NAME}...")
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
            print(f"  ✓ Re-ranker loaded.")
            logger.info(f"CrossEncoder loaded: {RERANKER_MODEL_NAME}")
        except ImportError:
            logger.warning("sentence-transformers not available. Skipping re-ranking.")
        except Exception as e:
            logger.error(f"Re-ranker load error: {e}")
    return _reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = RERANKER_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Re-rank documents using CrossEncoder for precise relevance scoring.

    Args:
        query: User question.
        documents: Merged candidate documents.
        top_k: Number of final documents to return.

    Returns:
        Top-K (Document, score) tuples sorted by re-rank score.
    """
    if not documents:
        return []

    reranker = get_reranker()

    if reranker is None:
        # Fallback: return first top_k docs without re-ranking
        logger.warning("Re-ranker unavailable. Returning top-K from merge order.")
        return [(doc, 1.0) for doc in documents[:top_k]]

    try:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = scored_docs[:top_k]
        logger.info(
            f"Re-ranked {len(documents)} → {len(top_docs)} chunks. "
            f"Top score: {top_docs[0][1]:.4f}" if top_docs else "No results."
        )
        return top_docs

    except Exception as e:
        logger.error(f"Re-ranking error: {e}")
        return [(doc, 1.0) for doc in documents[:top_k]]


# ─── CONTEXT CONSTRUCTION ─────────────────────────────────────────────────────

def build_context(
    ranked_docs: List[Tuple[Document, float]],
    max_context_chars: int = 3000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Construct a numbered context string from top-ranked chunks.
    Enforces a character limit to stay within SLM context window.

    Returns:
        Tuple of:
          - context_str: Formatted multi-chunk context for the SLM prompt.
          - citations: List of source metadata dicts for display.
    """
    context_parts = []
    citations = []
    total_chars = 0

    for i, (doc, score) in enumerate(ranked_docs, 1):
        chunk_text = doc.page_content.strip()
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        if total_chars + len(chunk_text) > max_context_chars:
            # Truncate last chunk to fit
            remaining = max_context_chars - total_chars
            if remaining > 100:  # Only add if meaningful
                chunk_text = chunk_text[:remaining] + "..."
            else:
                break

        context_part = (
            f"[Chunk {i} | Source: {source} | Page: {page}]\n"
            f"{chunk_text}"
        )
        context_parts.append(context_part)
        citations.append({
            "chunk_id": i,
            "source": source,
            "page": page,
            "score": round(float(score), 4),
        })

        total_chars += len(chunk_text)

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, citations


# ─── FULL RAG PIPELINE ────────────────────────────────────────────────────────

def run_rag_pipeline(
    query: str,
    vectorstore: FAISS,
    all_documents: Optional[List[Document]] = None,
    slm=None,
) -> Dict[str, Any]:
    """
    Execute the complete RAG pipeline end-to-end.

    Args:
        query:          User's natural language question.
        vectorstore:    Loaded FAISS index for the selected domain.
        all_documents:  Optional full document list for BM25 search.
                        If None, BM25 is skipped.
        slm:            SLMHandler instance for generation.

    Returns:
        Dictionary with:
          - answer (str)
          - citations (List[dict])
          - retrieved_count (int)
          - reranked_count (int)
    """
    from backend.slm_handler import get_slm

    if slm is None:
        slm = get_slm()

    print("\n  🔍 Step 1/4: Semantic search (FAISS)...")
    semantic_results = semantic_search(query, vectorstore, top_k=FAISS_TOP_K)

    print(f"  🔍 Step 2/4: Keyword search (BM25)...")
    if all_documents:
        bm25_results = bm25_search(query, all_documents, top_k=BM25_TOP_K)
    else:
        bm25_results = []
        logger.info("Skipping BM25: no document list provided.")

    print(f"  🔀 Step 3/4: Merging & deduplicating results...")
    merged_docs = merge_results(semantic_results, bm25_results)

    if not merged_docs:
        return {
            "answer": "Not found in provided documents.",
            "citations": [],
            "retrieved_count": 0,
            "reranked_count": 0,
        }

    print(f"  📊 Step 4/4: Re-ranking with CrossEncoder...")
    ranked_docs = rerank_documents(query, merged_docs, top_k=RERANKER_TOP_K)

    context_str, citations = build_context(ranked_docs)

    print(f"\n  🤖 Generating answer with SLM...")
    answer = slm.generate_answer(question=query, context=context_str)

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_count": len(merged_docs),
        "reranked_count": len(ranked_docs),
    }


# ─── DOCUMENT CACHE FOR BM25 ──────────────────────────────────────────────────

_document_cache: Dict[str, List[Document]] = {}


def cache_documents(domain: str, documents: List[Document]):
    """Cache a domain's documents in memory for BM25 reuse."""
    _document_cache[domain] = documents
    logger.info(f"Cached {len(documents)} documents for domain '{domain}'.")


def get_cached_documents(domain: str) -> Optional[List[Document]]:
    """Retrieve cached documents for BM25 search."""
    return _document_cache.get(domain)
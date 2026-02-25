"""Hybrid search: combine FAISS vector similarity with BM25 keyword scoring.

Fusion strategy: Reciprocal Rank Fusion (RRF) or weighted sum (configurable).
Default: weighted sum with α=0.7 (semantic) and β=0.3 (keyword).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from rag.embedder import embed_query
from rag.keyword_search import KeywordSearcher
from rag.vector_store import search as faiss_search

logger = logging.getLogger(__name__)

# ── fusion helpers ───────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    semantic_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge two ranked lists using RRF: score = Σ 1/(k + rank).

    *k* is the RRF constant (not top-k).  Returns merged list sorted by
    combined score, descending.
    """
    combined: Dict[str, Dict[str, Any]] = {}

    for rank, rec in enumerate(semantic_results, start=1):
        code = rec["course_code"]
        if code not in combined:
            combined[code] = dict(rec)
            combined[code].setdefault("keyword_score", 0.0)
        combined[code]["rrf_score"] = combined[code].get("rrf_score", 0.0) + 1.0 / (k + rank)

    for rank, rec in enumerate(keyword_results, start=1):
        code = rec["course_code"]
        if code not in combined:
            combined[code] = dict(rec)
            combined[code].setdefault("similarity_score", 0.0)
        combined[code].setdefault("keyword_score", rec.get("keyword_score", 0.0))
        combined[code]["rrf_score"] = combined[code].get("rrf_score", 0.0) + 1.0 / (k + rank)

    results = sorted(combined.values(), key=lambda r: r.get("rrf_score", 0.0), reverse=True)
    # Normalise rrf_score → final_score (0-1)
    if results:
        max_rrf = results[0].get("rrf_score", 1.0)
        for r in results:
            r["final_score"] = r.pop("rrf_score", 0.0) / max_rrf if max_rrf else 0.0
    return results


def _weighted_sum(
    semantic_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    alpha: float = 0.7,
    beta: float = 0.3,
) -> List[Dict[str, Any]]:
    """Merge by weighted sum: final = α·similarity + β·keyword."""
    combined: Dict[str, Dict[str, Any]] = {}

    for rec in semantic_results:
        code = rec["course_code"]
        if code not in combined:
            combined[code] = dict(rec)
            combined[code].setdefault("keyword_score", 0.0)
        combined[code]["similarity_score"] = rec.get("similarity_score", 0.0)

    for rec in keyword_results:
        code = rec["course_code"]
        if code not in combined:
            combined[code] = dict(rec)
            combined[code].setdefault("similarity_score", 0.0)
        combined[code]["keyword_score"] = rec.get("keyword_score", 0.0)

    for rec in combined.values():
        rec["final_score"] = (
            alpha * rec.get("similarity_score", 0.0)
            + beta * rec.get("keyword_score", 0.0)
        )

    return sorted(combined.values(), key=lambda r: r["final_score"], reverse=True)


# ── public API ───────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    index,
    metadata: List[Dict[str, Any]],
    keyword_searcher: KeywordSearcher,
    top_k: int = 5,
    department: Optional[str] = None,
    source: Optional[str] = None,
    fusion: str = "weighted",  # "weighted" | "rrf"
    alpha: float = 0.7,
    beta: float = 0.3,
    semantic_pool: int = 50,
    keyword_pool: int = 50,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Run hybrid vector + keyword search and return top-k merged results.

    Parameters
    ----------
    query : str
        Natural-language or code query.
    index : faiss.Index
    metadata : list of record dicts (parallel to index rows).
    keyword_searcher : KeywordSearcher instance.
    top_k : int
        Number of results to return.
    department / source : optional filters.
    fusion : ``"weighted"`` (default) or ``"rrf"``.
    alpha, beta : weights for weighted-sum fusion.
    semantic_pool, keyword_pool : how many candidates each leg retrieves.
    """
    # Step 1: vector search
    embed_kwargs: dict = {}
    if model_name:
        embed_kwargs["model_name"] = model_name
    q_vec = embed_query(query, **embed_kwargs)
    sem_results = faiss_search(
        index, metadata, q_vec, top_k=semantic_pool,
        department=department, source=source,
    )

    # Step 2: keyword search
    kw_results = keyword_searcher.search(
        query, top_k=keyword_pool, department=department, source=source,
    )

    # Step 3: fusion
    if fusion == "rrf":
        merged = _reciprocal_rank_fusion(sem_results, kw_results)
    else:
        merged = _weighted_sum(sem_results, kw_results, alpha=alpha, beta=beta)

    # Ensure every result has all three score fields
    for r in merged:
        r.setdefault("similarity_score", 0.0)
        r.setdefault("keyword_score", 0.0)
        r.setdefault("final_score", 0.0)

    logger.info(
        "Hybrid search (%s): %d semantic + %d keyword → %d merged → returning top %d",
        fusion, len(sem_results), len(kw_results), len(merged), top_k,
    )
    return merged[:top_k]


def vector_search(
    query: str,
    index,
    metadata: List[Dict[str, Any]],
    top_k: int = 5,
    department: Optional[str] = None,
    source: Optional[str] = None,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Pure vector search (no keyword component).

    Returns results with ``similarity_score``, ``keyword_score=0``,
    and ``final_score = similarity_score``.
    """
    embed_kwargs: dict = {}
    if model_name:
        embed_kwargs["model_name"] = model_name
    q_vec = embed_query(query, **embed_kwargs)
    results = faiss_search(
        index, metadata, q_vec, top_k=top_k,
        department=department, source=source,
    )
    for r in results:
        r.setdefault("keyword_score", 0.0)
        r["final_score"] = r.get("similarity_score", 0.0)
    return results

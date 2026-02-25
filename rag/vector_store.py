"""FAISS vector store – build, persist, load, and search.

Uses IndexFlatIP (inner-product) on L2-normalised vectors so that
inner-product == cosine similarity.

Persistence
-----------
- ``data/faiss.index``  – the FAISS binary index
- ``data/metadata.pkl`` – parallel list of record dicts (same order as index)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from rag.embedder import build_embed_text, embed_texts

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
COURSES_PATH = DATA_DIR / "courses.json"
FAISS_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "metadata.pkl"


class FAISSIndexMissingError(FileNotFoundError):
    """Raised when the FAISS index file does not exist on disk."""


# ── build / persist ──────────────────────────────────────────────────

def build_index(
    records: List[Dict[str, Any]],
    model_name: str | None = None,
) -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """Embed *records* and create a FAISS IndexFlatIP.

    Returns ``(index, metadata)`` where *metadata* is a list of record
    dicts in the same order as the index rows.
    """
    t0 = time.time()
    logger.info("Building FAISS index for %d records …", len(records))

    # 1. build dense text for every record
    texts = [build_embed_text(r) for r in records]

    # 2. embed
    kwargs: dict = {}
    if model_name:
        kwargs["model_name"] = model_name
    embeddings = embed_texts(texts, **kwargs)  # (N, dim), float32, L2-normed

    # 3. build FAISS IndexFlatIP (cosine via inner product on normed vecs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    elapsed = time.time() - t0
    logger.info(
        "FAISS index built: %d vectors, dim=%d (%.1f s)", index.ntotal, dim, elapsed
    )
    return index, records


def save_index(index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
    """Persist the FAISS index and metadata to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved FAISS index → %s", FAISS_PATH)
    logger.info("Saved metadata   → %s", META_PATH)


def load_index() -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load a previously persisted index + metadata.

    Raises ``FAISSIndexMissingError`` if files don't exist.
    """
    if not FAISS_PATH.exists():
        raise FAISSIndexMissingError(
            f"FAISS index not found at {FAISS_PATH}. "
            "Run `python run_rag.py --build` to create it."
        )
    if not META_PATH.exists():
        raise FAISSIndexMissingError(
            f"Metadata file not found at {META_PATH}. "
            "Run `python run_rag.py --build` to create it."
        )
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    logger.info("Loaded FAISS index (%d vectors) from %s", index.ntotal, FAISS_PATH)
    return index, metadata


def index_is_stale() -> bool:
    """Return True if courses.json is newer than the persisted index."""
    if not FAISS_PATH.exists() or not META_PATH.exists():
        return True
    courses_mtime = os.path.getmtime(COURSES_PATH)
    index_mtime = min(os.path.getmtime(FAISS_PATH), os.path.getmtime(META_PATH))
    return courses_mtime > index_mtime


def load_or_build(
    model_name: str | None = None,
    force_rebuild: bool = False,
) -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load existing index or rebuild if stale / forced.

    Returns ``(index, metadata)``.
    """
    if not force_rebuild and FAISS_PATH.exists() and not index_is_stale():
        return load_index()

    # (re)build
    if not COURSES_PATH.exists():
        raise FileNotFoundError(
            f"Course data not found at {COURSES_PATH}. "
            "Run the ETL pipeline first: python run_pipeline.py"
        )
    with open(COURSES_PATH, encoding="utf-8") as f:
        records = json.load(f)
    index, meta = build_index(records, model_name=model_name)
    save_index(index, meta)
    return index, meta


# ── search ───────────────────────────────────────────────────────────

def search(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    query_vec: np.ndarray,
    top_k: int = 50,
    department: str | None = None,
    source: str | None = None,
) -> List[Dict[str, Any]]:
    """Search the FAISS index and return scored result dicts.

    Supports optional post-filtering by *department* (case-insensitive
    substring) and *source* (exact match "CAB" / "BULLETIN").

    Each returned dict is a copy of the metadata record with an added
    ``similarity_score`` field (float 0-1).
    """
    # Retrieve more candidates when filtering so we can still fill top_k
    fetch_k = top_k * 3 if (department or source) else top_k
    fetch_k = min(fetch_k, index.ntotal)

    scores, ids = index.search(query_vec, fetch_k)
    scores = scores[0]  # shape (fetch_k,)
    ids = ids[0]

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores, ids):
        if idx == -1:
            continue
        rec = dict(metadata[idx])
        # Apply filters
        if department:
            if department.lower() not in rec.get("department", "").lower():
                continue
        if source:
            if rec.get("source", "").upper() != source.upper():
                continue
        rec["similarity_score"] = float(np.clip(score, 0.0, 1.0))
        results.append(rec)
        if len(results) >= top_k:
            break

    return results

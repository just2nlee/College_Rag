"""BM25 keyword search over the course corpus.

Uses rank_bm25.BM25Okapi for sparse scoring.  The corpus is tokenised
on whitespace + simple punctuation stripping so that short exact queries
like "CSCI0320" match well.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from rag.embedder import build_embed_text

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class KeywordSearcher:
    """Wraps BM25 over a list of course-record dicts."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records
        # Build corpus from the same dense text used for embedding
        corpus = [_tokenise(build_embed_text(r)) for r in records]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built over %d documents", len(records))

    def search(
        self,
        query: str,
        top_k: int = 50,
        department: str | None = None,
        source: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-k records scored by BM25 keyword relevance.

        Each returned dict is a copy with an added ``keyword_score`` field
        (float, normalised to 0-1).
        """
        tokens = _tokenise(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Normalise to 0-1
        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        normed = raw_scores / max_score

        # Pair with index, sort descending
        scored = sorted(enumerate(normed), key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in scored:
            if score <= 0:
                break
            rec = dict(self._records[idx])

            # Apply filters
            if department:
                if department.lower() not in rec.get("department", "").lower():
                    continue
            if source:
                if rec.get("source", "").upper() != source.upper():
                    continue

            rec["keyword_score"] = float(score)
            results.append(rec)
            if len(results) >= top_k:
                break

        return results

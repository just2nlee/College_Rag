"""RAG pipeline orchestrator – ties together index, search, and generation.

Typical usage
-------------
    from rag.pipeline import RAGPipeline

    rag = RAGPipeline()               # loads/builds FAISS index
    result = rag.query("What intro CS courses are available?")
    print(result["answer"])
    for rec in result["records"]:
        print(rec["course_code"], rec["final_score"])
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.generator import generate_answer
from rag.hybrid import hybrid_search, vector_search
from rag.keyword_search import KeywordSearcher
from rag.vector_store import (
    FAISSIndexMissingError,
    load_or_build,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
COURSES_PATH = DATA_DIR / "courses.json"


@dataclass
class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    On instantiation the FAISS index is loaded (or rebuilt if stale).
    Call ``.query()`` to search + generate.
    """

    model_name: Optional[str] = None
    force_rebuild: bool = False

    # populated by __post_init__
    _index: Any = field(default=None, repr=False, init=False)
    _metadata: List[Dict[str, Any]] = field(default_factory=list, repr=False, init=False)
    _keyword_searcher: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        """Load FAISS index + build BM25 keyword searcher."""
        self._index, self._metadata = load_or_build(
            model_name=self.model_name,
            force_rebuild=self.force_rebuild,
        )
        self._keyword_searcher = KeywordSearcher(self._metadata)
        logger.info(
            "RAGPipeline ready: %d vectors, BM25 over %d docs",
            self._index.ntotal,
            len(self._metadata),
        )

    def rebuild(self) -> None:
        """Force-rebuild the FAISS index from courses.json."""
        self.force_rebuild = True
        self._load()
        self.force_rebuild = False

    # ── query ────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        department: Optional[str] = None,
        source: Optional[str] = None,
        mode: str = "hybrid",  # "hybrid" | "vector"
        fusion: str = "weighted",  # "weighted" | "rrf"
        alpha: float = 0.7,
        beta: float = 0.3,
        generate: bool = True,
    ) -> Dict[str, Any]:
        """Run a query and return results + optional generated answer.

        Parameters
        ----------
        question : str
            Natural-language question or course-code query.
        top_k : int
            Number of results to return (default 5).
        department : str, optional
            Case-insensitive substring filter on department.
        source : str, optional
            Exact source filter ("CAB" or "BULLETIN").
        mode : str
            ``"hybrid"`` (default) or ``"vector"`` (semantic only).
        fusion : str
            ``"weighted"`` (default) or ``"rrf"`` for hybrid mode.
        alpha, beta : float
            Weights for weighted-sum fusion.
        generate : bool
            Whether to call the LLM to produce an answer (default True).

        Returns
        -------
        dict with keys:
            ``question``, ``answer``, ``context``, ``records``
        """
        t0 = time.time()
        logger.info("Query: %r  (mode=%s, top_k=%d)", question, mode, top_k)

        # Retrieve
        if mode == "vector":
            records = vector_search(
                question,
                self._index,
                self._metadata,
                top_k=top_k,
                department=department,
                source=source,
                model_name=self.model_name,
            )
        else:
            records = hybrid_search(
                question,
                self._index,
                self._metadata,
                self._keyword_searcher,
                top_k=top_k,
                department=department,
                source=source,
                fusion=fusion,
                alpha=alpha,
                beta=beta,
                model_name=self.model_name,
            )

        # Generate
        answer = ""
        context = ""
        if generate and records:
            answer, context = generate_answer(question, records)
        elif records:
            from rag.generator import assemble_context
            context = assemble_context(records)

        elapsed = time.time() - t0
        logger.info(
            "Query completed in %.2f s – %d results returned", elapsed, len(records)
        )

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "records": records,
        }

    # ── convenience ──────────────────────────────────────────────────

    @property
    def num_records(self) -> int:
        return len(self._metadata)

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index else 0

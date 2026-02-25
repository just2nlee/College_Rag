"""FastAPI backend for the Brown University Course RAG pipeline.

Start with:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config as cfg
from rag.pipeline import RAGPipeline
from rag.vector_store import FAISSIndexMissingError

# ── Logging setup ────────────────────────────────────────────────────

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Stdout handler
_stream = logging.StreamHandler()
_stream.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(_stream)

# NDJSON rotating file handler (only for query log lines, not startup msgs)
_log_dir = Path(cfg.LOG_PATH).parent
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = RotatingFileHandler(
    cfg.LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(message)s"))  # raw NDJSON

# Separate logger just for the NDJSON lines – keeps log file clean
_ndjson_logger = logging.getLogger("app.ndjson")
_ndjson_logger.setLevel(logging.INFO)
_ndjson_logger.addHandler(_file_handler)
_ndjson_logger.propagate = False


def _log_ndjson(entry: Dict[str, Any]) -> None:
    """Write a single NDJSON log line to the queries.log file."""
    _ndjson_logger.info(json.dumps(entry, default=str))


# ── In-memory query log for /evaluate ────────────────────────────────

_MAX_LOG_ENTRIES = 500
_query_log: deque[Dict[str, Any]] = deque(maxlen=_MAX_LOG_ENTRIES)

# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Brown University Course RAG API",
    version="1.0.0",
    description="Search and ask questions about Brown University courses.",
)

# Load the RAG pipeline at startup (lazy via lifespan would be ideal,
# but a module-level singleton keeps things simple and meets the <5 s
# startup requirement since the FAISS index is already built on disk).
_rag: Optional[RAGPipeline] = None


@app.on_event("startup")
async def _startup() -> None:
    global _rag
    try:
        logger.info("Loading RAG pipeline …")
        _rag = RAGPipeline(model_name=cfg.EMBEDDING_MODEL)
        logger.info("RAG pipeline ready — %d records indexed", _rag.index_size)
    except FAISSIndexMissingError as exc:
        logger.error("FAISS index not found: %s", exc)
        # _rag stays None → /query will return 503
    except Exception as exc:
        logger.error("Failed to load RAG pipeline: %s", exc)


# ── Request / response models ────────────────────────────────────────


class QueryRequest(BaseModel):
    q: Optional[str] = Field(None, description="Natural-language query or course code")
    department: Optional[str] = Field(None, description="Department filter (case-insensitive substring)")
    source: Optional[str] = Field(None, description='Source filter: "CAB" or "BULLETIN"')
    k: int = Field(default=cfg.DEFAULT_K, ge=1, le=50, description="Number of results")


class CourseResult(BaseModel):
    course_code: str
    title: str
    department: str
    similarity_score: float
    source: str
    instructor: Optional[str] = None
    meeting_times: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    results: List[CourseResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_size: int
    embedding_model: str


class ErrorResponse(BaseModel):
    error: str


class EvalEntry(BaseModel):
    timestamp: str
    query: str
    department_filter: Optional[str]
    latency_ms: float
    num_results: int


class EvalResponse(BaseModel):
    total_queries: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    recent: List[EvalEntry]


# ── Endpoints ─────────────────────────────────────────────────────────


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def post_query(body: QueryRequest):
    # Validate required field
    if not body.q or not body.q.strip():
        raise HTTPException(status_code=400, detail="q field is required")

    # Check pipeline readiness
    if _rag is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run python run_pipeline.py && python run_rag.py --build first.",
        )

    t0 = time.perf_counter()

    result = _rag.query(
        body.q.strip(),
        top_k=body.k,
        department=body.department,
        source=body.source,
        mode=cfg.SEARCH_MODE,
        fusion=cfg.FUSION_STRATEGY,
        alpha=cfg.FUSION_ALPHA,
        beta=cfg.FUSION_BETA,
        generate=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Build results list
    results = [
        CourseResult(
            course_code=r.get("course_code", ""),
            title=r.get("title", ""),
            department=r.get("department", ""),
            similarity_score=round(r.get("similarity_score", 0.0), 4),
            source=r.get("source", ""),
            instructor=r.get("instructor"),
            meeting_times=r.get("meeting_times"),
        )
        for r in result.get("records", [])
    ]

    # Log to NDJSON file + in-memory buffer
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": body.q.strip(),
        "department_filter": body.department,
        "latency_ms": round(latency_ms, 2),
        "num_results": len(results),
    }
    _log_ndjson(log_entry)
    _query_log.append(log_entry)

    logger.info(
        "POST /query q=%r dept=%s → %d results in %.0f ms",
        body.q, body.department, len(results), latency_ms,
    )

    return QueryResponse(
        answer=result.get("answer", ""),
        results=results,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def get_health():
    return HealthResponse(
        status="ok" if _rag is not None else "unavailable",
        index_size=_rag.index_size if _rag else 0,
        embedding_model=cfg.EMBEDDING_MODEL,
    )


@app.get("/evaluate", response_model=EvalResponse)
async def get_evaluate():
    entries = list(_query_log)
    latencies = [e["latency_ms"] for e in entries]

    def _percentile(data: List[float], p: float) -> Optional[float]:
        if not data:
            return None
        sorted_d = sorted(data)
        k = (len(sorted_d) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_d) else f
        return round(sorted_d[f] + (k - f) * (sorted_d[c] - sorted_d[f]), 2)

    return EvalResponse(
        total_queries=len(entries),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        recent=[
            EvalEntry(
                timestamp=e["timestamp"],
                query=e["query"],
                department_filter=e.get("department_filter"),
                latency_ms=e["latency_ms"],
                num_results=e["num_results"],
            )
            for e in entries[-20:]  # last 20
        ],
    )


# ── Error handlers to match spec format ──────────────────────────────


@app.exception_handler(HTTPException)
async def _http_exc_handler(request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

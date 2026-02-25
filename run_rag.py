#!/usr/bin/env python3
"""CLI entry-point for the RAG pipeline.

Usage
-----
    # Build / rebuild the FAISS index from data/courses.json
    python run_rag.py --build

    # Run a query (index must exist)
    python run_rag.py "What intro CS courses are available?"

    # Query with filters
    python run_rag.py "machine learning" --department "Computer Science" --top-k 10

    # Pure vector search (no keyword component)
    python run_rag.py "CSCI0320" --mode vector

    # Use RRF fusion instead of weighted sum
    python run_rag.py "software engineering" --fusion rrf

    # Skip LLM generation (retrieval only)
    python run_rag.py "linear algebra" --no-generate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Brown University course RAG pipeline",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural-language question or course code to search for.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build (or rebuild) the FAISS index from data/courses.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=None,
        help="Filter by department (case-insensitive substring match).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["CAB", "BULLETIN"],
        default=None,
        help="Filter by data source.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "vector"],
        default="hybrid",
        help="Search mode: hybrid (default) or vector-only.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["weighted", "rrf"],
        default="weighted",
        help="Fusion strategy for hybrid mode (default: weighted).",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip LLM answer generation (retrieval only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    args = parser.parse_args(argv)
    _configure_logging()

    from rag.pipeline import RAGPipeline

    if args.build:
        logging.getLogger(__name__).info("Building FAISS index …")
        RAGPipeline(force_rebuild=True)
        logging.getLogger(__name__).info("Index built successfully.")
        if not args.query:
            return 0

    if not args.query:
        parser.error("Please provide a query or use --build to create the index.")

    rag = RAGPipeline()
    result = rag.query(
        args.query,
        top_k=args.top_k,
        department=args.department,
        source=args.source,
        mode=args.mode,
        fusion=args.fusion,
        generate=not args.no_generate,
    )

    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        _pretty_print(result)

    return 0


def _pretty_print(result: dict) -> None:
    """Human-readable output."""
    print()
    print("=" * 70)
    print(f"  Question: {result['question']}")
    print("=" * 70)

    if result.get("answer"):
        print()
        print("Answer:")
        print("-" * 70)
        print(result["answer"])
        print("-" * 70)

    print()
    print(f"Top {len(result['records'])} results:")
    print()
    for i, rec in enumerate(result["records"], 1):
        sim = rec.get("similarity_score", 0.0)
        kw = rec.get("keyword_score", 0.0)
        final = rec.get("final_score", 0.0)
        print(
            f"  {i}. [{rec.get('course_code', '?')}] {rec.get('title', '?')}"
        )
        print(
            f"     Dept: {rec.get('department', '?')}  |  Source: {rec.get('source', '?')}"
        )
        print(
            f"     Scores — sim: {sim:.3f}  kw: {kw:.3f}  final: {final:.3f}"
        )
        desc = (rec.get("description") or "")[:120]
        if desc:
            print(f"     {desc}{'…' if len(rec.get('description', '')) > 120 else ''}")
        print()


if __name__ == "__main__":
    sys.exit(main())

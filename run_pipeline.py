#!/usr/bin/env python3
"""CLI entry-point for the Brown University course ETL pipeline.

Usage
-----
    python run_pipeline.py              # full run → data/courses.json + .csv
    python run_pipeline.py --dry-run    # scrape + normalise, log counts only
"""

from __future__ import annotations

import argparse
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
        description="Brown University course data ETL pipeline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and normalise but do not write output files.",
    )
    args = parser.parse_args(argv)
    _configure_logging()

    from etl.pipeline import run_pipeline

    records = run_pipeline(dry_run=args.dry_run)
    if not records:
        logging.getLogger(__name__).warning("Pipeline produced 0 records")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

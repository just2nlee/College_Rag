"""Main ETL pipeline – orchestrates scraping, normalisation, and output."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import List

from etl.models import CourseRecord, records_to_json
from etl.normalize import deduplicate, normalise_all
from etl.scraper_bulletin import scrape_bulletin
from etl.scraper_cab import scrape_cab

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
JSON_PATH = DATA_DIR / "courses.json"
CSV_PATH = DATA_DIR / "courses.csv"

_CSV_COLS = [
    "course_code", "title", "instructor", "meeting_times",
    "prerequisites", "department", "description", "source", "text",
]


def _write_json(records: List[CourseRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(records_to_json(records), encoding="utf-8")
    logger.info("Wrote %d records → %s", len(records), path)


def _write_csv(records: List[CourseRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r.to_dict())
    logger.info("Wrote %d records → %s", len(records), path)


def run_pipeline(*, dry_run: bool = False) -> List[CourseRecord]:
    """Execute the full ETL pipeline.

    Parameters
    ----------
    dry_run : bool
        If True, scrape and normalise but skip writing files.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("ETL pipeline started  (dry_run=%s)", dry_run)
    logger.info("=" * 60)

    # ── Stage 1: scrape ──────────────────────────────────────────────
    logger.info("── Stage 1: Scraping ──")
    cab_records = scrape_cab()
    logger.info("CAB  → %d records", len(cab_records))

    bulletin_records = scrape_bulletin()
    logger.info("BULLETIN → %d records", len(bulletin_records))

    all_records = cab_records + bulletin_records
    logger.info("Total raw: %d", len(all_records))

    # ── Stage 2: normalise ───────────────────────────────────────────
    logger.info("── Stage 2: Normalising ──")
    all_records = normalise_all(all_records)
    logger.info("After normalisation: %d", len(all_records))

    # ── Stage 3: deduplicate ─────────────────────────────────────────
    logger.info("── Stage 3: Deduplicating ──")
    all_records = deduplicate(all_records)

    # ── summary stats ────────────────────────────────────────────────
    cab_n = sum(1 for r in all_records if r.source == "CAB")
    bul_n = sum(1 for r in all_records if r.source == "BULLETIN")
    logger.info("  CAB records:      %d", cab_n)
    logger.info("  BULLETIN records: %d", bul_n)
    logger.info("  Total valid:      %d", len(all_records))

    # ── Stage 4: write output ────────────────────────────────────────
    if dry_run:
        logger.info("── Dry run: skipping file writes ──")
        logger.info("Would write %d records to %s and %s", len(all_records), JSON_PATH, CSV_PATH)
    else:
        logger.info("── Stage 4: Writing output ──")
        _write_json(all_records, JSON_PATH)
        _write_csv(all_records, CSV_PATH)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("ETL finished in %.1f s  (%d records)", elapsed, len(all_records))
    logger.info("=" * 60)
    return all_records

"""Normalisation and deduplication for scraped course records."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import List

from etl.models import CourseRecord

logger = logging.getLogger(__name__)


def _collapse_whitespace(text: str | None) -> str | None:
    if text is None:
        return None
    result = re.sub(r"\s+", " ", text).strip()
    return result if result else None


def normalise_record(rec: CourseRecord) -> CourseRecord:
    """Apply field-level cleaning to a single record (mutates in-place)."""
    rec.course_code = re.sub(r"\s+", "", rec.course_code).upper().strip()
    rec.title = (_collapse_whitespace(rec.title) or "").strip()
    rec.department = (_collapse_whitespace(rec.department) or "").strip()
    rec.description = (_collapse_whitespace(rec.description) or "").strip()
    rec.instructor = _collapse_whitespace(rec.instructor)
    rec.meeting_times = _collapse_whitespace(rec.meeting_times)
    rec.prerequisites = _collapse_whitespace(rec.prerequisites)
    rec._build_text()
    return rec


def normalise_all(records: List[CourseRecord]) -> List[CourseRecord]:
    """Normalise every record; drop those missing required fields."""
    good: List[CourseRecord] = []
    dropped = 0
    for rec in records:
        normalise_record(rec)
        if rec.is_valid():
            good.append(rec)
        else:
            dropped += 1
    if dropped:
        logger.info("Dropped %d invalid records during normalisation", dropped)
    return good


def deduplicate(records: List[CourseRecord]) -> List[CourseRecord]:
    """Merge records sharing the same course_code.

    CAB values are preferred on conflict.  Non-null BULLETIN fields fill
    gaps (e.g. prerequisites).
    """
    by_code: defaultdict[str, List[CourseRecord]] = defaultdict(list)
    for rec in records:
        by_code[rec.course_code].append(rec)

    merged: List[CourseRecord] = []
    dup_count = 0
    for code, group in sorted(by_code.items()):
        if len(group) == 1:
            merged.append(group[0])
            continue
        dup_count += 1
        cab = [r for r in group if r.source == "CAB"]
        bul = [r for r in group if r.source == "BULLETIN"]
        base = cab[0] if cab else group[0]
        for other in (bul + cab[1:]):
            base = base.merge(other)
        merged.append(base)

    logger.info(
        "Dedup: %d in → %d out (%d merges)", len(records), len(merged), dup_count
    )
    return merged

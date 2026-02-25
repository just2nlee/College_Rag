"""Scraper for Courses @ Brown (CAB) – https://cab.brown.edu/

CAB is a JavaScript SPA backed by a FOSE JSON API.  We call the API
directly – one bulk search request for the listing, then batched detail
requests for descriptions.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from etl.models import CourseRecord, department_for_code

logger = logging.getLogger(__name__)

BASE_URL = "https://cab.brown.edu"
API_URL = f"{BASE_URL}/api/?page=fose"

# Browser-like headers are required – plain UA gets HTTP 202 empty body
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://cab.brown.edu/",
}

# Semester codes to try in order (YYYYTT: 20=Spring, 15=Fall, 10=Summer)
_SEMESTER_CODES = ["202620", "202520", "202610", "202510"]

# Polite-scraping delay range (seconds)
_DELAY_MIN = 0.5
_DELAY_MAX = 1.0

# Maximum number of detail requests (controls runtime)
MAX_DETAIL_FETCHES = 300


def _delay() -> None:
    time.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


# ── helpers ──────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if "<" in text:
        text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _normalise_code(code: str) -> str:
    return re.sub(r"\s+", "", code).upper().strip()


# ── FOSE API calls ──────────────────────────────────────────────────

def _discover_semester(sess: requests.Session) -> Optional[str]:
    """Try each candidate semester and return the first with results."""
    for code in _SEMESTER_CODES:
        try:
            resp = sess.post(
                f"{API_URL}&route=search",
                json={
                    "other": {"srcdb": code},
                    "criteria": [{"field": "is_ind_study", "value": "N"}],
                },
                timeout=30,
            )
            if resp.ok and resp.text and "results" in resp.text:
                data = resp.json()
                n = data.get("count", 0)
                if n > 0:
                    logger.info("Discovered semester %s (%d courses)", code, n)
                    return code
        except Exception as exc:
            logger.debug("Semester %s probe failed: %s", code, exc)
        _delay()
    return None


def _fetch_search_results(sess: requests.Session, srcdb: str) -> List[Dict[str, Any]]:
    """One bulk request returns *all* course sections for the semester."""
    resp = sess.post(
        f"{API_URL}&route=search",
        json={
            "other": {"srcdb": srcdb},
            "criteria": [{"field": "is_ind_study", "value": "N"}],
        },
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def _fetch_detail(
    sess: requests.Session, srcdb: str, code: str, crn: str
) -> Dict[str, Any]:
    """Fetch the detail payload for one course section."""
    resp = sess.post(
        f"{API_URL}&route=details",
        json={
            "group": f"code:{code}",
            "key": f"crn:{crn}",
            "srcdb": srcdb,
            "matched": f"crn:{crn}",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── record conversion ───────────────────────────────────────────────

def _search_to_record(raw: Dict[str, Any]) -> Optional[CourseRecord]:
    """Convert a search-API dict to a CourseRecord (no description yet)."""
    code = _normalise_code(raw.get("code", ""))
    title = (raw.get("title") or "").strip()
    if not code or not title:
        return None

    instr = (raw.get("instr") or "").strip() or None
    meets = _strip_html(raw.get("meets") or "") or None

    return CourseRecord(
        course_code=code,
        title=title,
        instructor=instr,
        meeting_times=meets,
        prerequisites=None,
        department=department_for_code(code),
        description="",           # filled later via detail API
        source="CAB",
    )


def _enrich_with_detail(rec: CourseRecord, detail: Dict[str, Any]) -> None:
    """Mutate *rec* in-place with data from the detail payload."""
    desc = _strip_html(detail.get("description", ""))
    if desc:
        rec.description = desc

    meeting = _strip_html(detail.get("meeting_html", ""))
    if meeting and not rec.meeting_times:
        rec.meeting_times = meeting

    prereq = _strip_html(detail.get("prereqs", detail.get("prerequisites", "")))
    if prereq:
        rec.prerequisites = prereq

    rec._build_text()


# ── public API ───────────────────────────────────────────────────────

def scrape_cab(max_details: int = MAX_DETAIL_FETCHES) -> List[CourseRecord]:
    """Scrape courses from the CAB FOSE API.

    Returns only records that have a populated description.
    """
    sess = _session()

    # 1) discover active semester
    srcdb = _discover_semester(sess)
    if not srcdb:
        logger.warning("Could not discover an active CAB semester – returning 0 records")
        return []

    # 2) bulk search
    raw_results = _fetch_search_results(sess, srcdb)
    logger.info("CAB search returned %d section rows", len(raw_results))

    # 3) deduplicate by course code (keep first non-cancelled section)
    seen_codes: dict[str, Dict[str, Any]] = {}
    for r in raw_results:
        code = _normalise_code(r.get("code", ""))
        if not code:
            continue
        if r.get("isCancelled") == "1":
            continue
        if code not in seen_codes:
            seen_codes[code] = r
    logger.info("Unique active course codes: %d", len(seen_codes))

    # 4) convert to CourseRecords
    records_by_code: dict[str, CourseRecord] = {}
    for code, raw in seen_codes.items():
        rec = _search_to_record(raw)
        if rec:
            records_by_code[code] = rec

    # 5) batch-fetch details (limited to max_details)
    codes_needing_detail = list(records_by_code.keys())[:max_details]
    logger.info(
        "Fetching details for %d / %d courses …",
        len(codes_needing_detail),
        len(records_by_code),
    )
    for i, code in enumerate(codes_needing_detail):
        raw = seen_codes[code]
        crn = raw.get("crn", "")
        if not crn:
            continue
        try:
            detail = _fetch_detail(sess, srcdb, raw.get("code", ""), crn)
            if "fatal" in detail:
                logger.debug("Detail API error for %s: %s", code, detail["fatal"])
                continue
            _enrich_with_detail(records_by_code[code], detail)
        except Exception as exc:
            logger.debug("Detail fetch failed for %s: %s", code, exc)
        _delay()
        if (i + 1) % 50 == 0:
            logger.info("  … fetched %d / %d details", i + 1, len(codes_needing_detail))

    # 6) keep only records with descriptions
    final = [r for r in records_by_code.values() if r.description]
    logger.info("CAB scraper finished: %d records with descriptions", len(final))
    return final

"""Scraper for Brown University Bulletin – https://bulletin.brown.edu/

Strategy
--------
1.  Crawl the concentrations index page to discover department links.
2.  For each concentration page, extract ``/search/?P=CODE`` links that
    point to individual course descriptions.
3.  Deduplicate those links across all departments.
4.  Fetch each unique ``/search/?P=CODE`` page and parse the
    ``.courseblock`` that contains the full description and offering
    table.
"""

from __future__ import annotations

import logging
import random
import re
import time
import urllib.parse
from typing import Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup

from etl.models import CourseRecord, department_for_code

logger = logging.getLogger(__name__)

BASE_URL = "https://bulletin.brown.edu"
CONCENTRATIONS_URL = f"{BASE_URL}/the-college/concentrations/"
USER_AGENT = "BrownCollegeRAG-ETL/1.0 (academic project; polite scraper)"

_DELAY_MIN = 0.5
_DELAY_MAX = 1.0

# Maximum course-detail pages to fetch (controls runtime)
MAX_COURSE_FETCHES = 250

_CODE_RE = re.compile(r"([A-Z]{2,6})\s*(\d{4}[A-Z]?)")


def _delay() -> None:
    time.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


# ── step 1: discover concentration (department) pages ────────────────

def _discover_concentration_links(sess: requests.Session) -> List[Dict[str, str]]:
    """Return [{name, url}, …] for every concentration listed on the index."""
    resp = sess.get(CONCENTRATIONS_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    links: List[Dict[str, str]] = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if href.startswith("/the-college/concentrations/") and href != "/the-college/concentrations/" and text:
            links.append({"name": text, "url": f"{BASE_URL}{href}"})

    # deduplicate by URL
    seen: Set[str] = set()
    unique: List[Dict[str, str]] = []
    for d in links:
        if d["url"] not in seen:
            seen.add(d["url"])
            unique.append(d)
    logger.info("Found %d concentration pages", len(unique))
    return unique


# ── step 2: collect /search/?P= links from concentration pages ──────

def _collect_course_links(
    sess: requests.Session, dept_pages: List[Dict[str, str]]
) -> Dict[str, str]:
    """Return {course_code: search_url} across all concentration pages."""
    code_to_url: Dict[str, str] = {}
    for i, dept in enumerate(dept_pages):
        _delay()
        try:
            resp = sess.get(dept["url"], timeout=30)
            if not resp.ok:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if "/search/?P=" not in href:
                continue
            # extract course code from the P parameter
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            raw_code = qs.get("P", [""])[0]
            if not raw_code:
                continue
            code = re.sub(r"\s+", "", raw_code).upper()
            if not _CODE_RE.match(code):
                continue
            full_url = f"{BASE_URL}/search/?P={urllib.parse.quote(raw_code)}"
            code_to_url.setdefault(code, full_url)

        if (i + 1) % 20 == 0:
            logger.info(
                "  scanned %d / %d concentration pages (%d unique codes so far)",
                i + 1, len(dept_pages), len(code_to_url),
            )

    logger.info("Collected %d unique course codes from concentration pages", len(code_to_url))
    return code_to_url


# ── step 3: parse /search/?P= detail pages ──────────────────────────

def _parse_search_page(html: str) -> Optional[Dict[str, str]]:
    """Extract course data from a Bulletin search-result page.

    Expected structure (discovered empirically):
        <article class="search-courseresult">
          <h3>CSCI 0150.  Introduction to OOP …</h3>
          <div class="courseblock">
            <p class="courseblockdesc">…description…</p>
            <table class="sc_sctable tbl_offering">
              <tr>…semester, code, section, crn, days, time, instructor…</tr>
            </table>
          </div>
        </article>
    """
    soup = BeautifulSoup(html, "lxml")

    article = soup.select_one("article.search-courseresult")
    if not article:
        return None

    # title line: "CSCI 0150.  Introduction to …"
    h3 = article.select_one("h3")
    if not h3:
        return None
    h3_text = h3.get_text(strip=True)

    m = _CODE_RE.search(h3_text)
    if not m:
        return None
    code = re.sub(r"\s+", "", m.group(0)).upper()

    # everything after the code (strip leading dots / whitespace)
    title = h3_text[m.end():].strip().lstrip(".").strip()
    # remove trailing period
    title = title.rstrip(".")

    # description
    desc_el = article.select_one(".courseblockdesc")
    desc = desc_el.get_text(strip=True) if desc_el else ""

    # prerequisites (look for "Prerequisite:" in desc)
    prereq: Optional[str] = None
    if desc:
        pm = re.search(r"(?:Prerequisites?|Prereqs?):\s*(.+?)(?:\.\s|$)", desc, re.I)
        if pm:
            prereq = pm.group(1).strip()

    return {
        "code": code,
        "title": title,
        "description": desc,
        "prerequisites": prereq,
    }


# ── public API ───────────────────────────────────────────────────────

def scrape_bulletin(max_fetches: int = MAX_COURSE_FETCHES) -> List[CourseRecord]:
    """Scrape courses from the Brown Bulletin.

    Returns only records with a populated description.
    """
    sess = _session()

    # 1) discover department pages
    dept_pages = _discover_concentration_links(sess)
    if not dept_pages:
        logger.warning("No concentration pages found on Bulletin")
        return []

    # 2) collect /search/?P= links
    code_to_url = _collect_course_links(sess, dept_pages)
    if not code_to_url:
        logger.warning("No course links found on concentration pages")
        return []

    # 3) fetch detail pages (limited)
    codes = list(code_to_url.keys())[:max_fetches]
    logger.info("Fetching %d / %d course detail pages …", len(codes), len(code_to_url))

    records: List[CourseRecord] = []
    for i, code in enumerate(codes):
        _delay()
        url = code_to_url[code]
        try:
            resp = sess.get(url, timeout=30)
            if not resp.ok:
                continue
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", url, exc)
            continue

        parsed = _parse_search_page(resp.text)
        if not parsed or not parsed["description"]:
            continue

        try:
            rec = CourseRecord(
                course_code=parsed["code"],
                title=parsed["title"],
                instructor=None,
                meeting_times=None,
                prerequisites=parsed.get("prerequisites"),
                department=department_for_code(parsed["code"]),
                description=parsed["description"],
                source="BULLETIN",
            )
            records.append(rec)
        except Exception as exc:
            logger.debug("Record creation failed for %s: %s", code, exc)

        if (i + 1) % 50 == 0:
            logger.info("  … fetched %d / %d pages (%d records)", i + 1, len(codes), len(records))

    logger.info("Bulletin scraper finished: %d records with descriptions", len(records))
    return records

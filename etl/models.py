"""Unified course schema for the ETL pipeline."""

from __future__ import annotations

import dataclasses
import json
from typing import List, Optional

VALID_SOURCES = ("CAB", "BULLETIN")

# ── Dept-code → human-readable name mapping ──────────────────────────
DEPT_CODE_MAP: dict[str, str] = {
    "AFRI": "Africana Studies",
    "AMST": "American Studies",
    "ANTH": "Anthropology",
    "APMA": "Applied Mathematics",
    "ARAB": "Arabic Studies",
    "ARCH": "Architecture",
    "ASYR": "Assyriology",
    "BIOL": "Biology",
    "CATL": "Catalan",
    "CHEM": "Chemistry",
    "CHIN": "Chinese",
    "CLAS": "Classics",
    "CLPS": "Cognitive, Linguistic & Psychological Sciences",
    "COLT": "Comparative Literature",
    "CSCI": "Computer Science",
    "CZCH": "Czech",
    "DEVL": "Development Studies",
    "EAST": "East Asian Studies",
    "ECON": "Economics",
    "EDUC": "Education",
    "EGYT": "Egyptology",
    "ENGL": "English",
    "ENGN": "Engineering",
    "ENVS": "Environmental Studies",
    "ETHN": "Ethnic Studies",
    "FREN": "French Studies",
    "GEOL": "Earth, Environmental and Planetary Sciences",
    "GERM": "German Studies",
    "GNSS": "Gender and Sexuality Studies",
    "GREK": "Ancient Greek",
    "HIAA": "History of Art and Architecture",
    "HIST": "History",
    "HNDI": "Hindi-Urdu",
    "IAPA": "International and Public Affairs",
    "ITAL": "Italian Studies",
    "JAPN": "Japanese",
    "JUDS": "Judaic Studies",
    "KREA": "Korean",
    "LACA": "Latin American and Caribbean Studies",
    "LATN": "Latin",
    "LITR": "Literary Arts",
    "MATH": "Mathematics",
    "MCM": "Modern Culture and Media",
    "MGRK": "Modern Greek",
    "MES": "Middle East Studies",
    "MUSC": "Music",
    "NEUR": "Neuroscience",
    "PHIL": "Philosophy",
    "PHP": "Public Health",
    "PHYS": "Physics",
    "PLCY": "Public Policy",
    "PLME": "Program in Liberal Medical Education",
    "POBS": "Portuguese and Brazilian Studies",
    "POLS": "Political Science",
    "RELS": "Religious Studies",
    "RUSS": "Russian",
    "SAIL": "Science, Art, Innovation, and Leadership",
    "SIGN": "Sign Language",
    "SLAV": "Slavic Studies",
    "SOC": "Sociology",
    "SPAN": "Hispanic Studies",
    "SWED": "Swedish",
    "TAPS": "Theatre Arts and Performance Studies",
    "TKSH": "Turkish",
    "UNIV": "University Courses",
    "URBN": "Urban Studies",
    "VISA": "Visual Art",
}


def department_for_code(course_code: str) -> str:
    """Derive a human-readable department name from a course code prefix."""
    import re

    m = re.match(r"^([A-Z]{2,6})", course_code.upper())
    if not m:
        return ""
    prefix = m.group(1)
    return DEPT_CODE_MAP.get(prefix, prefix)


@dataclasses.dataclass
class CourseRecord:
    """A single normalised course record."""

    course_code: str
    title: str
    department: str
    description: str
    source: str  # "CAB" | "BULLETIN"
    instructor: Optional[str] = None
    meeting_times: Optional[str] = None
    prerequisites: Optional[str] = None
    text: Optional[str] = None  # concatenated field for Feature 2

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {self.source!r}")
        self._build_text()

    def _build_text(self) -> None:
        parts = [self.title or "", self.description or "", self.department or ""]
        self.text = " ".join(p.strip() for p in parts if p.strip())

    # ── serialisation ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CourseRecord":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    # ── merging (CAB preferred) ──────────────────────────────────────
    def merge(self, other: "CourseRecord") -> "CourseRecord":
        merged: dict = {}
        for f in dataclasses.fields(self):
            self_v = getattr(self, f.name)
            other_v = getattr(other, f.name)
            if f.name == "source":
                merged["source"] = "CAB" if "CAB" in (self.source, other.source) else self.source
            elif self_v is not None and str(self_v).strip():
                merged[f.name] = self_v
            else:
                merged[f.name] = other_v
        return CourseRecord(**merged)

    def is_valid(self) -> bool:
        return all([
            self.course_code and self.course_code.strip(),
            self.title and self.title.strip(),
            self.department and self.department.strip(),
            self.description and self.description.strip(),
            self.source in VALID_SOURCES,
        ])


def records_to_json(records: List[CourseRecord]) -> str:
    return json.dumps([r.to_dict() for r in records], indent=2, ensure_ascii=False)

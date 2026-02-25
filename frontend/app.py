"""Streamlit frontend for the Brown University Course RAG system.

Launch with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"
COURSES_PATH = Path(__file__).resolve().parent.parent / "data" / "courses.json"

# ── Page setup ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brown Course Search",
    page_icon="🐻",
    layout="wide",
)

# Compact CSS – keeps the page usable without scrolling
st.markdown(
    """
    <style>
    /* tighten spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 0.5rem; }
    h1 { margin-bottom: 0.2rem; }
    /* latency badge */
    .latency-badge {
        display: inline-block;
        background: #262730;
        color: #fafafa;
        padding: 0.2rem 0.55rem;
        border-radius: 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_departments() -> list[str]:
    """Return sorted unique departments from the courses JSON."""
    try:
        with open(COURSES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return sorted({r["department"] for r in data})
    except Exception:
        return []


def call_backend(
    query: str,
    department: str | None = None,
    source: str | None = None,
) -> dict:
    """POST to /query and return parsed JSON or raise."""
    payload: dict = {"q": query}
    if department:
        payload["department"] = department
    if source and source != "All":
        payload["source"] = source
    resp = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def check_health() -> dict | None:
    """GET /health – returns None on failure."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Filters")

    departments = load_departments()
    dept_options = ["All Departments"] + departments
    selected_dept = st.selectbox("Department", dept_options, index=0)

    selected_source = st.radio("Source", ["All", "CAB", "BULLETIN"], index=0)

    st.divider()

    # Health indicator
    health = check_health()
    if health and health.get("status") == "ok":
        st.success(
            f"Backend online — {health['index_size']} courses indexed",
            icon="✅",
        )
    else:
        st.error("Backend offline — start the API server first.", icon="🔴")

# ── Main area ────────────────────────────────────────────────────────

st.title("🐻 Brown University Course Search")
st.caption("Ask questions about courses using natural language.")

query = st.text_area(
    "Your question",
    placeholder='e.g. "What introductory CS courses are available?" or "CSCI1300"',
    height=80,
    label_visibility="collapsed",
)

submit = st.button("🔍  Search", type="primary", use_container_width=True)

# ── Query execution ──────────────────────────────────────────────────

if submit:
    if not query or not query.strip():
        st.warning("Please enter a question or course code.")
    else:
        dept_filter = None if selected_dept == "All Departments" else selected_dept
        try:
            with st.spinner("Searching courses & generating answer …"):
                result = call_backend(query.strip(), dept_filter, selected_source)

            # — Latency badge —
            latency = result.get("latency_ms", 0)
            st.markdown(
                f'<span class="latency-badge">⏱ {latency:,.0f} ms</span>',
                unsafe_allow_html=True,
            )

            # — Generated answer —
            answer = result.get("answer", "")
            if answer:
                st.subheader("💡 Answer")
                st.markdown(answer)
            else:
                st.info("No generated answer (LLM backend may be unavailable).")

            # — Results table —
            records = result.get("results", [])
            if records:
                st.subheader(f"📋 Results ({len(records)})")
                table_rows = [
                    {
                        "Rank": idx + 1,
                        "Code": r["course_code"],
                        "Title": r["title"],
                        "Department": r["department"],
                        "Instructor": r.get("instructor") or "—",
                        "Meeting Times": r.get("meeting_times") or "—",
                        "Source": r["source"],
                        "Similarity": round(r["similarity_score"], 4),
                    }
                    for idx, r in enumerate(records)
                ]
                st.dataframe(
                    table_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No matching courses found. Try broadening your query.")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot reach the backend. Make sure the API server is running:\n\n"
                "```\nuvicorn app:app --reload --port 8000\n```"
            )
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("error", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"❌ Backend error: {detail}")
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")

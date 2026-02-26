# Brown University Course RAG

Here I've built a Retrieval-Augmented Generation (RAG) system for exploring Brown University course offerings. It scrapes course data from CAB and the Bulletin, builds a hybrid semantic + keyword search index, exposes a FastAPI backend, and provides a Streamlit frontend for natural-language course queries by leveraging GPT-4o-mini.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ETL Pipeline                               │
│   run_pipeline.py                                                   │
│   ┌──────────────┐     ┌──────────────────┐    ┌─────────────────┐  │
│   │ scraper_cab  │     │ scraper_bulletin │    │  normalize.py   │  │
│   │  (JSON API)  │───▶│   (HTML/lxml)    │───▶│ unified schema  │  │
│   └──────────────┘     └──────────────────┘    └────────┬────────┘  │
│                                                         │           │
│                                                  data/courses.json  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Vector Store Build                          │
│   run_rag.py --build                                                │
│   ┌────────────────────────────┐    ┌───────────────────────────┐   │
│   │  embedder.py               │    │  keyword_search.py        │   │
│   │  BAAI/bge-base-en-v1.5     │    │  BM25Okapi                │   │
│   │  768-d, L2-normed          │    │  (rank-bm25)              │   │
│   └────────────┬───────────────┘    └──────────────┬────────────┘   │
│                │                                   │                │
│   data/faiss.index                    (in-memory, rebuilt on load)  │
│   data/metadata.pkl                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                             │
│   app.py  (uvicorn, port 8000)                                      │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐     │
│   │ POST /query │  │  GET /health │  │    GET /evaluate       │     │
│   │  hybrid.py  │  │ index stats  │  │  p50/p95/p99 latency   │     │ 
│   │ generator.py│  └──────────────┘  └────────────────────────┘     │
│   │(GPT-4o-mini │                                                   │
│   │ via OpenAI) │                                                   │
│   └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend                            │
│   frontend/app.py  (port 8501)                                      │
│   Query input · Department dropdown · Source filter                 │
│   LLM answer box · Results table · Latency badge                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ (tested on 3.14.0) |
| Streamlit | 1.30+ |
| OpenAI API key | GPT-4o-mini access required |
| Git | any recent version |

> **Note:** A CUDA-capable GPU is optional. The embedding model (`BAAI/bge-base-en-v1.5`) runs on CPU but is faster on GPU.

---

## Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/just2nlee/College_Rag.git
cd College_Rag

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key – create a .env file in the project root
echo OPENAI_API_KEY=sk-... > .env

# 5. Run the ETL pipeline (scrapes CAB + Bulletin → data/courses.json)
python run_pipeline.py

# 6. Build the FAISS vector index
python run_rag.py --build

# 7. Start the FastAPI backend (Terminal 1)
uvicorn app:app --reload --port 8000

# 8. Start the Streamlit frontend (Terminal 2)
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

---

## Environment Variable Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI key for GPT-4o-mini generation |
| `EMBEDDING_MODEL` | No | `BAAI/bge-base-en-v1.5` | Sentence-transformers model name |
| `DEFAULT_K` | No | `5` | Number of results to return |
| `INDEX_PATH` | No | `data/faiss.index` | Path to FAISS binary index |
| `METADATA_PATH` | No | `data/metadata.pkl` | Path to metadata pickle |
| `COURSES_PATH` | No | `data/courses.json` | Path to course data |
| `LOG_PATH` | No | `logs/queries.log` | Path to NDJSON query log |
| `SEARCH_MODE` | No | `hybrid` | `hybrid` or `vector` |
| `FUSION_STRATEGY` | No | `weighted` | `weighted` or `rrf` |
| `FUSION_ALPHA` | No | `0.7` | Semantic weight in weighted fusion |
| `FUSION_BETA` | No | `0.3` | Keyword weight in weighted fusion |
| `PORT` | No | `8000` | Backend server port |

All variables can also be set in `config.yaml` (env vars take precedence).

---

## Example Queries

### 1. Introductory programming courses
**Query:** `What introductory CS courses are available?`

**Expected answer excerpt:**
> CSCI0150 (Introduction to Object-Oriented Programming) and CSCI0111 (Computing Foundations: Data) are both entry-level courses requiring no prerequisites...

---

### 2. Machine learning
**Query:** `What are some machine learning courses that are available?`

**Expected answer excerpt:**
> BHDS2130 (Methods III: Statistical Machine Learning), CSCI1410 (Artificial Intelligence), and APMA2070 (Deep Learning for Scientists & Engineers) cover machine learning topics...

---

### 3. Schedule-aware query
**Query:** `List all CAB courses taught on Fridays after 3 pm related to machine learning`

**Expected answer excerpt:**
> Based on the provided courses, APMA2070 lists meeting times "M 3-5:30p"...
> The context does not confirm Friday-specific machine learning courses — only the schedules listed above are available.

---

## Adding a New Data Source

1. Create `etl/scraper_<source>.py` (e.g. `scraper_canvas.py`).
2. Implement a function `scrape() -> list[dict]` returning raw course records.
3. Ensure each record has at minimum: `course_code`, `title`, `department`, `description`.
4. Add optional fields as available: `prerequisites`, `instructor`, `meeting_times`.
5. Set the `source` field to a unique string (e.g. `"CANVAS"`).
6. In `etl/pipeline.py`, import your scraper and call `scrape()` alongside the existing ones.
7. Pass new records through `etl/normalize.py` → `normalize_record(raw, source="CANVAS")`.
8. Append the normalized records to the combined list before writing `data/courses.json`.
9. Re-run `python run_pipeline.py` then `python run_rag.py --build`.
10. Optionally add `"CANVAS"` to the source radio in `frontend/app.py` for an explicit filter.

---

## Design Decisions & Trade-offs

### Hybrid Search (Vector + BM25)
**Decision:** Combine two different search methods, semantic search (FAISS) and keyword search (BM25),and blend their results.

**Why two methods?** Think of it like having two librarians helping you find books:
- **Semantic search** understands *meaning*. If you ask "classes about AI," it knows to include courses about "machine learning" and "neural networks" even if those exact words aren't in your query.
- **Keyword search (BM25)** matches *exact words*. If you search for "CSCI0320," it finds that exact course code instantly.

Neither method alone is perfect. Semantic search might miss an exact course code you typed. Keyword search won't understand that "intro programming" means the same as "beginner coding." By combining both (70% semantic, 30% keyword), the system handles both types of queries well.

### Embedding Model: BAAI/bge-base-en-v1.5
**Decision:** Use a free, open-source AI model to convert course descriptions into numerical "fingerprints" (embeddings) that capture their meaning.

**Why this model?** Embeddings are how the system understands similarity. When you search "machine learning courses," the system compares your query's fingerprint against every course's fingerprint to find the closest matches.

- **Why `bge-base` specifically?** It's a well-tested model designed specifically for search tasks (not general chat), so it's better at finding relevant documents. It's also free and runs locally—no API calls needed.
- **Why not a bigger model?** Larger models (like `bge-large` or OpenAI's `text-embedding-3-large`) are slightly more accurate but take longer to process and use more memory. For ~500 courses, `bge-base` is plenty accurate and keeps searches fast (under 50ms).

### In-process FAISS Index
**Decision:** Load the index into memory at FastAPI startup rather than using a hosted vector DB.  
**Trade-off:** Zero infrastructure overhead, sub-millisecond search latency. Does not scale beyond ~1M documents without IVF indexing or a dedicated vector DB (Pinecone, Weaviate, Qdrant).

### LLM Backend: OpenAI GPT-4o-mini
**Decision:** Use OpenAI's GPT-4o-mini model exclusively for generating natural-language answers.

**Why GPT-4o-mini?** After the system finds relevant courses (via hybrid search), an LLM reads those results and writes a helpful, human-friendly answer. GPT-4o-mini is:
- **Fast:** Responses typically arrive in 1-2 seconds.
- **Affordable:** At ~$0.15 per million output tokens, a typical query costs a fraction of a cent.
- **Reliable:** OpenAI's API has high uptime and consistent quality.

**Why not local models?** Running an LLM locally (e.g., via Ollama) would eliminate API costs and keep data private, but requires a powerful GPU and more complex setup. For this project, the simplicity and reliability of OpenAI outweighed those benefits.

### Streamlit Frontend
**Decision:** Streamlit over React for minimal setup (no build step, no Node.js required).  
**Trade-off:** Streamlit re-runs the entire script on each interaction, making streaming responses harder to implement. A React + fetch frontend would be more appropriate for production.

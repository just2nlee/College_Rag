# Technical Report: Brown University Course RAG System

---

## 1. RAG Pipeline — End-to-End

The pipeline has four stages that run in sequence:

### 1.1 ETL (Extract-Transform-Load)
`run_pipeline.py` orchestrates two scrapers:

- **`etl/scraper_cab.py`** — hits the CAB FOSE JSON API (`/api/parameter/searchresults`) to extract 296 course records including `meeting_times`, `instructor`, and enrollment data.
- **`etl/scraper_bulletin.py`** — uses `requests` + `lxml` to scrape 208 course records from the Brown Course Bulletin HTML pages (department-by-department).

Both scrapers feed into `etl/normalize.py`, which maps heterogeneous raw fields to a unified schema (`course_code`, `title`, `department`, `description`, `prerequisites`, `instructor`, `meeting_times`, `source`). Deduplication by `course_code` is applied. The merged 504-record dataset is written to `data/courses.json`.

### 1.2 Indexing
`run_rag.py --build` triggers two parallel indexing paths:

- **Vector index:** `rag/embedder.py` constructs a rich text string per course (code + title + department + instructor + meeting times + description + prerequisites) and encodes it with `BAAI/bge-base-en-v1.5` (sentence-transformers). Embeddings are L2-normalised so that FAISS inner-product search equals cosine similarity. The resulting 504 × 768 float32 matrix is stored as `data/faiss.index`; the parallel metadata list is pickled to `data/metadata.pkl`.
- **Keyword index:** `rag/keyword_search.py` builds a `BM25Okapi` index over tokenised course text. This is rebuilt in memory each time the pipeline loads.

### 1.3 Hybrid Retrieval
At query time `rag/hybrid.py` runs both legs:

1. **Semantic leg:** embed the query → FAISS `IndexFlatIP.search()` → top-50 candidates with cosine scores.
2. **Keyword leg:** BM25 scoring over all corpus documents → top-50 candidates.
3. **Fusion:** default is weighted sum `final = 0.7 × semantic + 0.3 × keyword`. Reciprocal Rank Fusion (RRF) is also implemented and selectable via `--fusion rrf`.

Optional post-filters by `department` (substring) and `source` (exact) are applied after fusion, before slicing to `top_k`.

### 1.4 Generation
`rag/generator.py` assembles the top-k records into a structured context block (course code, title, department, source, instructor, meeting times, description, prerequisites). This context is inserted into a chat prompt with a system instruction to cite course codes and use schedule information when present. The prompt is sent to GPT-4o-mini (OpenAI). The function returns `(answer, context)`.

---

## 2. Embedding Model Choice

**Model:** `BAAI/bge-base-en-v1.5` (Beijing Academy of AI, base variant)

### Why this model
| Property | Value |
|---|---|
| Dimensions | 768 |
| Max sequence length | 512 tokens |
| MTEB retrieval rank (at selection time) | Top-10 open-source |
| Licence | MIT |
| Size | ~435 MB |

BGE models are trained with instruction-based fine-tuning specifically for retrieval tasks and consistently outperform general-purpose encoders (e.g. `all-MiniLM-L6-v2`) on passage retrieval benchmarks. The base variant was chosen over `bge-large-en-v1.5` (1024-d) to keep index build time and memory usage practical on CPU.

### Trade-offs vs alternatives

| Model | Dimensions | Quality | Latency (CPU) | Cost |
|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` ✅ | 768 | High | ~4 s / 504 docs | Free |
| `BAAI/bge-large-en-v1.5` | 1024 | Higher | ~10 s / 504 docs | Free |
| `text-embedding-3-small` | 1536 | Higher | API call | $0.02/1M tokens |
| `text-embedding-3-large` | 3072 | Highest | API call | $0.13/1M tokens |
| `all-MiniLM-L6-v2` | 384 | Moderate | ~1 s / 504 docs | Free |

For a student-facing tool with a small corpus (~500 docs), `bge-base` offers the best quality/cost/latency balance. An upgrade to `text-embedding-3-small` would be the recommended next step for production.

---

## 3. Performance Observations

### Latency

All numbers measured on a Windows laptop (Intel i7, no GPU, 16 GB RAM):

| Operation | Observed latency |
|---|---|
| Server startup (FAISS load + BM25 rebuild) | ~1.6 s |
| Hybrid search (query embed + FAISS + BM25 + fusion) | 40–80 ms |
| GPT-4o-mini generation (network round-trip) | 8,000–30,000 ms |
| End-to-end `/query` response (with generation) | 8–30 s |
| End-to-end `/query` response (retrieval-only) | <100 ms |

The dominant latency factor is the OpenAI API call. Retrieval itself is fast (<100 ms total) and would comfortably serve hundreds of concurrent users.

### Retrieval Quality
- **Exact course code lookups** (e.g. `CSCI0320`): BM25 correctly surfaces the exact match in position 1 in all observed tests.
- **Semantic queries** (e.g. `introductory programming`): Vector search ranks relevant courses accurately; hybrid fusion improves recall vs pure vector in ~70% of manual test cases.
- **Schedule queries** (e.g. `courses on Fridays after 3pm`): Improved after embedding `meeting_times` into the index text. LLM correctly filters by day/time when the field is populated. CAB records have schedule data; Bulletin records do not, which limits recall for cross-source schedule queries.

### Observed Failure Modes
1. **Missing schedule data in Bulletin records** — 208 Bulletin courses have `meeting_times = null`, so schedule-filtered queries only return CAB courses reliably.
2. **Token budget hallucination** — when the context fills the token budget and gets truncated mid-sentence, GPT-4o-mini occasionally fabricates completions. Mitigation: reduce `DEFAULT_K` or increase `max_tokens`.
3. **Department name mismatch** — CAB uses abbreviations (`CSCI`) while Bulletin uses full names (`Computer Science`), so a filter for `"CSCI"` misses Bulletin entries. Workaround: use the Streamlit dropdown which lists actual values.
4. **BM25 tokenisation on hyphenated codes** — `CSCI-0320` vs `CSCI0320` can cause a miss. The current normaliser strips hyphens at ETL time.

---

## 4. Production Improvements

### 4.1 Re-ranking with a Cross-Encoder
Replace the current bi-encoder retrieval with a two-stage pipeline: retrieve top-50 with the bi-encoder, then re-rank with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`). Cross-encoders attend jointly to query and document, substantially improving MRR at the cost of ~200 ms extra latency.

### 4.2 Semantic Caching
Cache `(query_embedding, top_k, department, source) → response` in Redis with a TTL of 1 hour. Repeated or near-duplicate queries (cosine similarity > 0.97) serve from cache, eliminating the OpenAI call and cutting p50 latency from ~15 s to <100 ms for cache hits.

### 4.3 Authentication & Rate Limiting
Add API key authentication (FastAPI `Security` dependency) and per-key rate limiting (e.g. `slowapi` middleware, 60 req/min). This prevents abuse of the OpenAI key and allows usage tracking per user.

### 4.4 Structured Observability
Replace the current NDJSON log file with OpenTelemetry traces exported to Grafana/Loki. Instrument: query latency, retrieval scores, LLM token usage, and error rates. Add a Prometheus `/metrics` endpoint so the `/evaluate` stats are scraped automatically.

### 4.5 Streaming LLM Responses
Switch the OpenAI call to streaming (`stream=True`) and use FastAPI's `StreamingResponse` + Server-Sent Events to push tokens to the Streamlit frontend as they arrive. This reduces perceived latency from ~15 s to a token-by-token stream starting in ~300 ms.

### 4.6 Incremental Index Updates
Currently, adding a new course requires a full rebuild. Implement an incremental update: compute embeddings only for new/changed records, append them to the FAISS index with `index.add()`, and update `metadata.pkl`. This reduces re-index time from ~30 s to <1 s for small changes.

### 4.7 IVF Index for Scale
`IndexFlatIP` performs exact search in O(N) time. At >100k documents, switch to `IndexIVFFlat` (inverted file index) which reduces search time to O(N/n_lists) with a small recall trade-off. At 504 documents this is unnecessary, but is the correct upgrade path.

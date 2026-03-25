# CSRAG — Corrective + Self-Reflective RAG with Memory

A production-grade conversational RAG API combining four advanced patterns into a single unified LangGraph pipeline, deployed on AWS EC2 with full CI/CD.

---

## Graph Architecture

![CSRAG Graph](assets/csrag_graph_design.svg)

---

## What's inside

| Pattern | What it does |
|---|---|
| **CRAG** | Scores every retrieved chunk (0–1). CORRECT → use internal docs. AMBIGUOUS/INCORRECT → rewrite query + Tavily web search. |
| **SRAG** | Verifies factual grounding (`fully_supported` / `partially_supported` / `no_support`) and usefulness (`useful` / `not_useful`) of every answer — including direct answers. Revises if needed. |
| **STM** | Rolling conversation summarisation via LangGraph `AsyncPostgresSaver`. Summarises when message count exceeds threshold; keeps only last 2 messages + summary. |
| **LTM** | Persistent per-user fact store via LangGraph `AsyncPostgresStore`. Extracts atomic facts from every user message; injects them into every system prompt. |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | Ollama `nomic-embed-text` (local, 768-dim) |
| Vector DB | Qdrant Cloud |
| Memory DB | PostgreSQL (`AsyncPostgresStore` + `AsyncPostgresSaver`) |
| Web Search | Tavily |
| Framework | LangGraph + FastAPI |
| Config | pydantic-settings |
| Registry | Docker Hub |
| Hosting | AWS EC2 (t3.small + 4GB swap) |

---

## Pipeline Graph Flow

```
START
  │
  ▼
ltm_remember           ← extract user facts from message · store to Postgres · read LTM context
  │
  ▼
decide_retrieval       ← LLM classifier: private doc question? or general knowledge?
  │
  ├─ False ──────────────────────────────────────────────────────────────────────┐
  │  (general knowledge)                                                          │
  │                                                                               ▼
  │                                                                    generate_direct
  │                                                                    (LLM answers from training)
  │                                                                               │
  └─ True ──────────────────┐                                                    │
     (document question)    │                                                    │
                             ▼                                                    │
                        retrieve_docs  ← Qdrant similarity search · top-k chunks │
                             │                                                    │
                             ▼                                                    │
                        evaluate_docs  (CRAG)                                    │
                        score every chunk 0.0–1.0 · asyncio.gather (parallel)   │
                             │                                                    │
               CORRECT ──────┤  AMBIGUOUS/INCORRECT                             │
                 │           └──→ rewrite_query → web_search ──┐                │
                 │                                              │                │
                 └──────────────────────────────────────────────▼                │
                                                          refine_context         │
                                                  (batch sentence filter · 1 LLM call)
                                                               │                 │
                                                               ▼                 │
                                                         generate_answer         │
                                                  (RAG prompt · context only)    │
                                                               │                 │
                                                               ▼                 │
                                                         verify_support (SRAG)   │
                                                  grounded in context?           │
                                                               │                 │
                                          fully_supported ─────┤                 │
                                                │               └──→ revise_answer ↺ max 2×
                                                │                    (remove unsupported claims)
                                                │
                                                └──────────────────────────────────┘
                                                                │
                                                                ▼
                                                      verify_usefulness (SRAG)   ← SHARED by both paths
                                                      does answer help the user?
                                                                │
                                              useful ───────────┤   not_useful
                                                │               └──→ rewrite_question → retrieve_docs ↺ max 2×
                                                ▼
                                         stm_summarize  ← compress conversation if messages > threshold
                                                │
                                               END
```

### Key design decisions

| Decision | Reason |
|---|---|
| `decide_retrieval` defaults to `False` (not `True`) on error or uncertainty | Wasteful to run full retrieval pipeline for questions the LLM already knows. Worst case: answer is slightly less grounded — caught by `verify_usefulness`. |
| `generate_direct` routes to `verify_usefulness`, not `stm_summarize` | Direct answers are now quality-checked. If the LLM's direct answer is `not_useful`, the pipeline falls back to full retrieval + CRAG + SRAG automatically. |
| `verify_usefulness` is shared by both direct and RAG paths | Single quality gate for all answers regardless of how they were generated. |
| `verify_support` skipped on direct path (`issup = "skipped"`) | No retrieved context exists to verify grounding against. Checking support with an empty context always returns `no_support`, which is meaningless. |
| `delete_collection` immediately recreates empty collection | Prevents the app from entering a broken state where `search()` fails on a missing collection. |
| `search()` self-heals on empty/missing collection | Qdrant raises on empty collections. Catch → `_ensure_collection()` → return `[]` → CRAG scores 0 → INCORRECT → web search. No crash. |
| CRAG sentence filtering is batched (1 LLM call) | Original was N serial calls (one per sentence). Reduced latency from ~30–60s to ~1–2s for typical documents. |
| All LLM calls use `ainvoke` (async) | Non-blocking. All nodes are `async def`. Qdrant and Tavily (sync SDKs) are offloaded to thread pool via `run_in_executor`. |
| CRAG doc scoring uses `asyncio.gather` | All N document evaluations run concurrently, not serially. |

---

## Project structure

```
CSRAG/
├── app/
│   ├── __init__.py                # __version__
│   ├── main.py                    # FastAPI app · async lifespan · router registration
│   ├── config.py                  # All settings via pydantic-settings
│   │
│   ├── api/
│   │   ├── schemas.py             # All Pydantic request/response models
│   │   └── routes/
│   │       ├── health.py          # GET /health · GET /health/ready
│   │       ├── documents.py       # POST /documents/upload · GET /documents/info · DELETE /documents/collection
│   │       ├── chat.py            # POST /chat · POST /chat/stream · GET /chat/history/{thread_id}
│   │       └── memory.py          # GET /memory/{user_id} · DELETE /memory/{user_id}
│   │
│   ├── core/
│   │   ├── embeddings.py          # OllamaEmbeddings (nomic-embed-text) — lru_cache singleton
│   │   ├── vector_store.py        # VectorStoreService (Qdrant Cloud) — self-healing search + auto-recreate on delete
│   │   ├── document_processor.py  # PDF/TXT/CSV loading + chunking
│   │   ├── csrag_engine.py        # Main orchestration service (wraps the graph)
│   │   │
│   │   ├── graph/
│   │   │   ├── state.py           # CSRAGState TypedDict
│   │   │   ├── nodes.py           # All async node functions + routing functions
│   │   │   └── builder.py         # StateGraph wiring + compile()
│   │   │
│   │   ├── crag/
│   │   │   ├── evaluator.py       # CRAGEvaluator — parallel chunk scoring · CORRECT/AMBIGUOUS/INCORRECT
│   │   │   └── web_search.py      # WebSearchService — async query rewrite + Tavily
│   │   │
│   │   ├── srag/
│   │   │   └── verifier.py        # SRAGVerifier — support check · usefulness check · answer revision
│   │   │
│   │   └── memory/
│   │       ├── stm.py             # STMSummarizer — async conversation compression
│   │       └── ltm.py             # LTMService — async Postgres fact extraction + storage
│   │
│   └── utils/
│       └── logger.py              # setup_logging() · get_logger()
│
├── tests/
│   ├── conftest.py                # Shared fixtures (mock lifespan, AsyncMock stores, sync TestClient)
│   ├── test_health.py             # 9 tests: /health, /health/ready, /, /docs, /openapi.json
│   ├── test_chat.py               # 25 tests: /chat, /chat/stream, /chat/history, validation
│   ├── test_documents.py          # 21 tests: DocumentProcessor unit + /documents endpoints
│   ├── test_memory.py             # 22 tests: STMSummarizer, LTMService, /memory endpoints
│   ├── test_crag.py               # 7 tests: CRAGEvaluator (all async)
│   ├── test_srag.py               # 8 tests: SRAGVerifier (all async)
│   └── test_config.py             # 19 tests: Settings defaults + env var loading
│
├── assets/
│   └── csrag_graph_design.svg     # LangGraph pipeline diagram (auto-generated)
│
├── infra/
│   └── MANUAL_STEPS.md            # EC2 setup guide
│
├── .github/workflows/cicd.yml     # CI/CD: test → build → deploy
├── requirements.txt               # All deps with version bounds
├── .env.example
├── Dockerfile
├── docker-compose.yml             # Local dev (Postgres only)
└── pytest.ini                     # asyncio_mode = auto
```

---

## CI/CD Pipeline

Every `git push origin main` automatically:

```
Unit Tests (pytest ~2 min)
       ↓
Docker Build + Smoke Test (~4 min)
       ↓
Push to Docker Hub + Deploy to EC2 (~2 min)
```

GitHub Actions secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `EC2_HOST`, `EC2_USERNAME`, `EC2_SSH_KEY`, `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `TAVILY_API_KEY`.

---

## Local Development

### Prerequisites

1. **Ollama** — install from https://ollama.com and pull the embedding model:

```bash
ollama pull nomic-embed-text
```

2. **PostgreSQL** — use the provided `docker-compose.yml`:

```bash
docker-compose up postgres -d
```

3. **External API keys**:

| Service | Where to get credentials |
|---|---|
| Groq | https://console.groq.com |
| Qdrant Cloud | https://cloud.qdrant.io |
| Tavily | https://app.tavily.com |

### Setup

```bash
# 1. Clone / navigate to the project
cd CSRAG

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux
# Edit .env and fill in your API keys

# 5. Start Postgres
docker-compose up postgres -d

# 6. Run the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Run tests

```bash
pytest tests/ -v --asyncio-mode=auto
```

---

## API Endpoints

### Health
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe — no external deps checked |
| GET | `/health/ready` | Readiness probe — checks Qdrant + Postgres |

### Documents
| Method | Path | Description |
|---|---|---|
| POST | `/documents/upload` | Upload PDF, TXT, or CSV · chunk + embed + store in Qdrant |
| GET | `/documents/info` | Collection metadata (name, point count, status) |
| DELETE | `/documents/collection` | Delete all indexed documents · auto-recreates empty collection |

### Chat
| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Full CSRAG pipeline query — returns answer + all metadata |
| POST | `/chat/stream` | Streaming answer token by token (text/plain) |
| GET | `/chat/history/{thread_id}` | Retrieve full message history + STM summary for a thread |

### Memory
| Method | Path | Description |
|---|---|---|
| GET | `/memory/{user_id}` | List all LTM facts stored for a user |
| DELETE | `/memory/{user_id}` | Clear all LTM facts for a user (irreversible) |

---

## Example — chat request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "thread_id": "thread-001",
    "user_id": "user-001",
    "include_sources": true
  }'
```

### Response fields

| Field | Description |
|---|---|
| `answer` | The generated answer |
| `sources` | Internal (Qdrant) + web (Tavily) documents used |
| `crag_verdict` | `CORRECT` / `AMBIGUOUS` / `INCORRECT` — or `""` if retrieval was skipped |
| `crag_reason` | Verdict justification with max chunk score |
| `issup` | `fully_supported` / `partially_supported` / `no_support` / `skipped` |
| `evidence` | Direct quotes from context supporting the answer |
| `isuse` | `useful` / `not_useful` |
| `use_reason` | Justification for the usefulness verdict |
| `retries` | SRAG answer revision loop counter (max 2) |
| `rewrite_tries` | Question rewrite loop counter (max 2) |
| `processing_time_ms` | End-to-end latency |

> **Note on `issup = "skipped"`:** When `decide_retrieval` returns `False`, there is no retrieved context to verify grounding against. The answer goes directly through `verify_usefulness` and `issup` is `"skipped"` — this is expected and correct.

---

## Example — retrieve chat history

```bash
curl http://localhost:8000/chat/history/thread-001
```

```json
{
  "thread_id": "thread-001",
  "messages": [
    {"role": "human",     "content": "What is the refund policy?"},
    {"role": "assistant", "content": "The refund window is 30 days..."},
    {"role": "human",     "content": "Can I get a partial refund?"},
    {"role": "assistant", "content": "Yes, partial refunds are available..."}
  ],
  "summary": "",
  "message_count": 4
}
```

If STM has compressed the conversation, `messages` contains only the most recent 2 messages and `summary` holds the compressed history of all earlier turns.

---

## Routing logic — when does web search fire?

| Question type | `need_retrieval` | CRAG verdict | Web search? |
|---|---|---|---|
| `"Who is PM of India?"` | `False` | — | ❌ Never |
| `"What is Python?"` | `False` | — | ❌ Never |
| `"What is our refund policy?"` (docs uploaded) | `True` | `CORRECT` | ❌ Not needed |
| `"What is our refund policy?"` (no docs) | `True` | `INCORRECT` | ✅ Fired |
| `"What is our Q3 revenue?"` (partial docs) | `True` | `AMBIGUOUS` | ✅ Supplements |

Web search only fires when: retrieval was attempted AND CRAG found internal documents inadequate.

---

## EC2 Deployment (Production)

The app runs on **AWS EC2 t3.small** with:
- 4GB swap space (required for Ollama on 2GB RAM)
- Ollama running natively on the host (bound to `0.0.0.0`)
- CSRAG container on Docker network `csrag-net`
- Postgres container on Docker network `csrag-net`
- `nomic-embed-text` model (274MB, fits on t3.small)

See `infra/MANUAL_STEPS.md` for the complete one-time EC2 setup guide. All subsequent deployments are automated via GitHub Actions.

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `QDRANT_URL` | — | Qdrant Cloud cluster URL (required) |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key (required) |
| `TAVILY_API_KEY` | — | Tavily web search key (required) |
| `POSTGRES_URI` | `postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable` | Postgres connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (`http://172.17.0.1:11434` on EC2) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_DIMENSION` | `768` | Vector dimension (must match model) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature |
| `CRAG_UPPER_THRESHOLD` | `0.7` | Chunk score at or above → CORRECT |
| `CRAG_LOWER_THRESHOLD` | `0.3` | All chunks below → INCORRECT |
| `SRAG_MAX_RETRIES` | `2` | Max answer revision loops |
| `MAX_REWRITE_TRIES` | `2` | Max question rewrite loops |
| `STM_MESSAGE_THRESHOLD` | `6` | Summarise when message count exceeds this |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve from Qdrant |
| `CHUNK_SIZE` | `900` | Token chunk size for document splitting |
| `CHUNK_OVERLAP` | `150` | Token overlap between chunks |
| `TAVILY_MAX_RESULTS` | `5` | Max Tavily search results |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |

---

## Changelog

### Latest improvements

| Change | File | Impact |
|---|---|---|
| `generate_direct` now routes to `verify_usefulness` instead of `stm_summarize` | `builder.py` | All answers — direct and RAG — are quality-checked. Poor direct answers fall back to full retrieval automatically. |
| `decide_retrieval` prompt rewritten with 8 concrete examples + default changed from `True` to `False` | `nodes.py` | General knowledge questions ("Who is PM of India?") no longer trigger unnecessary retrieval + web search. |
| `delete_collection()` recreates empty collection immediately after deletion | `vector_store.py` | App never enters broken state after deletion. Uploads work instantly. |
| `search()` self-heals on empty/missing collection | `vector_store.py` | No crash when searching an empty collection. Returns `[]` → CRAG returns `INCORRECT` → web search fires. |
| `issup = "skipped"` added as valid Literal in `CSRAGState` and `ChatResponse` | `state.py`, `schemas.py` | Honest reporting — direct answers bypass support verification which has no context to check against. |
| CRAG scoring parallelised with `asyncio.gather` | `evaluator.py` | All N chunk evaluations run concurrently. |
| Sentence filtering batched (single LLM call instead of N) | `nodes.py` | Latency reduced from ~30–60s to ~1–2s per query. |
| All nodes converted to `async def` with `ainvoke` | `nodes.py`, `srag/`, `crag/`, `memory/` | No event loop blocking. Full async pipeline. |
| Sync Qdrant + Tavily calls offloaded to `run_in_executor` | `nodes.py`, `web_search.py`, `documents.py` | Thread pool prevents event loop stalls. |

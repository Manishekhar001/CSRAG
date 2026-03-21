# CSRAG — Corrective + Self-Reflective RAG with Short & Long Term Memory

A production-grade conversational RAG API that combines four advanced patterns
into a single unified LangGraph pipeline.

---

## What's inside

| Pattern | What it does |
|---|---|
| **CRAG** | Scores every retrieved chunk (0–1). CORRECT → use internal docs. AMBIGUOUS/INCORRECT → rewrite query + Tavily web search. |
| **SRAG** | Verifies factual grounding (`fully_supported` / `partially_supported` / `no_support`) and usefulness of every answer. Revises if needed. |
| **STM** | Rolling conversation summarisation via LangGraph `PostgresSaver`. Summarises when message count exceeds threshold; keeps only last 2 messages + summary. |
| **LTM** | Persistent per-user fact store via LangGraph `PostgresStore`. Extracts atomic facts from every user message; injects them into every system prompt. |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | Ollama `mxbai-embed-large` (local) |
| Vector DB | Qdrant Cloud |
| Memory DB | PostgreSQL (LTM store + STM checkpointer) |
| Web Search | Tavily |
| Framework | LangGraph + FastAPI |
| Config | pydantic-settings |

---

## Project structure

```
CSRAG/
├── app/
│   ├── __init__.py            # __version__
│   ├── main.py                # FastAPI app, lifespan, router registration
│   ├── config.py              # All settings via pydantic-settings
│   │
│   ├── api/
│   │   ├── schemas.py         # All Pydantic request/response models
│   │   └── routes/
│   │       ├── health.py      # GET /health, GET /health/ready
│   │       ├── documents.py   # POST /documents/upload, GET /documents/info, DELETE /documents/collection
│   │       ├── chat.py        # POST /chat, POST /chat/stream
│   │       └── memory.py      # GET /memory/{user_id}, DELETE /memory/{user_id}
│   │
│   ├── core/
│   │   ├── embeddings.py      # OllamaEmbeddings (mxbai-embed-large) — cached
│   │   ├── vector_store.py    # VectorStoreService (Qdrant Cloud)
│   │   ├── document_processor.py  # PDF/TXT/CSV loading + chunking
│   │   ├── csrag_engine.py    # Main orchestration service (wraps the graph)
│   │   │
│   │   ├── graph/
│   │   │   ├── state.py       # CSRAGState TypedDict
│   │   │   ├── nodes.py       # All 14 node functions + routing functions
│   │   │   └── builder.py     # StateGraph wiring + compile()
│   │   │
│   │   ├── crag/
│   │   │   ├── evaluator.py   # CRAGEvaluator — chunk scoring + CORRECT/AMBIGUOUS/INCORRECT
│   │   │   └── web_search.py  # WebSearchService — query rewrite + Tavily
│   │   │
│   │   ├── srag/
│   │   │   └── verifier.py    # SRAGVerifier — support check + usefulness check + revision
│   │   │
│   │   └── memory/
│   │       ├── stm.py         # STMSummarizer — conversation compression
│   │       └── ltm.py         # LTMService — Postgres fact extraction + storage
│   │
│   └── utils/
│       └── logger.py          # setup_logging(), get_logger(), LoggerMixin
│
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Prerequisites

### 1. Ollama (local embeddings)

```bash
# Install Ollama from https://ollama.com then pull the embedding model
ollama pull mxbai-embed-large
```

### 2. PostgreSQL

Use the provided `docker-compose.yml` (runs Postgres on port 5442):

```bash
docker-compose up postgres -d
```

### 3. External services

| Service | Where to get credentials |
|---|---|
| Groq API key | https://console.groq.com |
| Qdrant Cloud URL + key | https://cloud.qdrant.io |
| Tavily API key | https://app.tavily.com |

---

## Setup

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

# 5. Start Postgres (if not already running)
docker-compose up postgres -d

# 6. Run the API
python -m app.main
# or
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

### Health
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks Qdrant + Postgres) |

### Documents
| Method | Path | Description |
|---|---|---|
| POST | `/documents/upload` | Upload PDF, TXT, or CSV |
| GET | `/documents/info` | Collection metadata |
| DELETE | `/documents/collection` | Delete all indexed documents |

### Chat
| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Full CSRAG pipeline query |
| POST | `/chat/stream` | Streaming answer |

### Memory
| Method | Path | Description |
|---|---|---|
| GET | `/memory/{user_id}` | List all LTM facts for a user |
| DELETE | `/memory/{user_id}` | Clear all LTM facts for a user |

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

Response includes:
- `answer` — the generated answer
- `sources` — internal (Qdrant) + web (Tavily) documents used
- `crag_verdict` — CORRECT / AMBIGUOUS / INCORRECT
- `issup` — fully_supported / partially_supported / no_support
- `evidence` — direct quotes from the context
- `isuse` — useful / not_useful
- `retries` / `rewrite_tries` — loop iteration counters

---

## Graph flow

```
START
  → ltm_remember          (extract + store user facts in Postgres)
  → decide_retrieval       (needs docs? yes/no)
      → generate_direct    (no retrieval — general knowledge)
      → retrieve_docs      (Qdrant similarity search)
          → evaluate_docs  (CRAG — score chunks)
              CORRECT  → refine_context
              else     → rewrite_query → web_search → refine_context
          → generate_answer
          → verify_support (SRAG — is it grounded?)
              not fully supported → revise_answer ↺ (max 2x)
          → verify_usefulness (SRAG — does it help?)
              not_useful → rewrite_question → retrieve_docs ↺ (max 2x)
  → stm_summarize          (compress if messages > 6)
END
```

---

## Docker (full stack)

```bash
# Build and run everything (Postgres + CSRAG API)
docker-compose up --build
```

> Note: Ollama must be running on the host machine. Set  
> `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`  
> when running inside Docker on Mac/Windows.

---

## Environment variables reference

See `.env.example` for the full list with descriptions.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key |
| `QDRANT_URL` | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `TAVILY_API_KEY` | — | Tavily web search key |
| `POSTGRES_URI` | `postgresql://...localhost:5442/...` | Postgres connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `CRAG_UPPER_THRESHOLD` | `0.7` | Score above this → CORRECT |
| `CRAG_LOWER_THRESHOLD` | `0.3` | All below this → INCORRECT |
| `SRAG_MAX_RETRIES` | `2` | Max answer revision loops |
| `MAX_REWRITE_TRIES` | `2` | Max question rewrite loops |
| `STM_MESSAGE_THRESHOLD` | `6` | Summarise when messages exceed this |

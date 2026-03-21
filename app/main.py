"""FastAPI application entry point.

Startup sequence:
  1. Load .env → Settings
  2. Setup logging
  3. Connect Qdrant (VectorStoreService)
  4. Open PostgresStore (LTM)
  5. Open PostgresSaver (STM checkpointer)
  6. Build + compile CSRAG graph (CSRAGEngine)
  7. Store all shared services on app.state
  8. On shutdown: close Postgres connections cleanly
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from app import __version__
from app.api.routes import chat, documents, health, memory
from app.config import get_settings
from app.core.csrag_engine import CSRAGEngine
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger, setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")

    # ------------------------------------------------------------------
    # 1. Qdrant vector store
    # ------------------------------------------------------------------
    logger.info("Initialising VectorStoreService (Qdrant)...")
    app.state.vector_store = VectorStoreService()
    logger.info("VectorStoreService ready")

    # ------------------------------------------------------------------
    # 2. PostgresStore (Long-Term Memory)
    # ------------------------------------------------------------------
    logger.info("Connecting PostgresStore (LTM)...")
    store = PostgresStore.from_conn_string(settings.postgres_uri)
    store.__enter__()
    store.setup()  # idempotent — creates tables if not present
    app.state.store = store
    logger.info("PostgresStore (LTM) ready")

    # ------------------------------------------------------------------
    # 3. PostgresSaver (Short-Term Memory checkpointer)
    # ------------------------------------------------------------------
    logger.info("Connecting PostgresSaver (STM checkpointer)...")
    checkpointer = PostgresSaver.from_conn_string(settings.postgres_uri)
    checkpointer.__enter__()
    checkpointer.setup()  # idempotent
    app.state.checkpointer = checkpointer
    logger.info("PostgresSaver (STM checkpointer) ready")

    # ------------------------------------------------------------------
    # 4. CSRAG Engine (compiles the full LangGraph graph)
    # ------------------------------------------------------------------
    logger.info("Compiling CSRAG graph...")
    app.state.engine = CSRAGEngine(
        vector_store=app.state.vector_store,
        store=store,
        checkpointer=checkpointer,
    )
    logger.info("CSRAG Engine ready — all services online")

    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("Shutting down — closing Postgres connections...")
    try:
        checkpointer.__exit__(None, None, None)
    except Exception as e:
        logger.warning(f"Checkpointer close error: {e}")
    try:
        store.__exit__(None, None, None)
    except Exception as e:
        logger.warning(f"Store close error: {e}")
    logger.info("Shutdown complete")


# ----------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description="""
## CSRAG — Corrective + Self-Reflective RAG with Memory

A production-grade conversational RAG system combining:

- **CRAG** (Corrective RAG) — scores retrieved chunks; falls back to web search for AMBIGUOUS/INCORRECT retrievals
- **SRAG** (Self-Reflective RAG) — verifies factual grounding and usefulness of every answer; revises if needed
- **STM** (Short-Term Memory) — rolling conversation summarisation via LangGraph checkpointing (Postgres)
- **LTM** (Long-Term Memory) — persistent user fact store via LangGraph PostgresStore

### Stack
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: Ollama (mxbai-embed-large, local)
- **Vector DB**: Qdrant Cloud
- **Memory DB**: PostgreSQL
- **Web Search**: Tavily
- **Framework**: LangGraph + FastAPI
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------
# CORS middleware
# ----------------------------------------------------------------------
_raw_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
allowed_origins: list[str] = _raw_origins if _raw_origins != ["*"] else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(memory.router)


# ----------------------------------------------------------------------
# Root
# ----------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — service info."""
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs",
    }


# ----------------------------------------------------------------------
# Global exception handler
# ----------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — logs and returns a structured 500."""
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )


# ----------------------------------------------------------------------
# Dev entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

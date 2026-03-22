from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app import __version__
from app.api.routes import chat, documents, health, memory
from app.config import get_settings
from app.core.csrag_engine import CSRAGEngine
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger, setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")

    logger.info("Initialising VectorStoreService (Qdrant)...")
    app.state.vector_store = VectorStoreService()
    logger.info("VectorStoreService ready")

    logger.info("Connecting AsyncPostgresStore (LTM)...")
    async with await AsyncPostgresStore.from_conn_string(settings.postgres_uri) as store:
        await store.setup()
        app.state.store = store
        logger.info("AsyncPostgresStore (LTM) ready")

        logger.info("Connecting AsyncPostgresSaver (STM checkpointer)...")
        async with await AsyncPostgresSaver.from_conn_string(settings.postgres_uri) as checkpointer:
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            logger.info("AsyncPostgresSaver (STM checkpointer) ready")

            logger.info("Compiling CSRAG graph...")
            app.state.engine = CSRAGEngine(
                vector_store=app.state.vector_store,
                store=store,
                checkpointer=checkpointer,
            )
            logger.info("CSRAG Engine ready — all services online")

            yield

    logger.info("Shutdown complete")


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
- **Embeddings**: Ollama (nomic-embed-text, local)
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

_raw_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
allowed_origins: list[str] = _raw_origins if _raw_origins != ["*"] else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(memory.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

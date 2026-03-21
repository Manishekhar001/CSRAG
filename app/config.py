"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Groq (LLM provider)
    # ------------------------------------------------------------------
    groq_api_key: str

    # Main chat / reasoning model (llama-3.3-70b-versatile)
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # Dedicated model for LTM memory extraction (stronger / more precise)
    memory_llm_model: str = "llama-3.3-70b-versatile"
    memory_llm_temperature: float = 0.0

    # ------------------------------------------------------------------
    # Ollama (local embeddings — mxbai-embed-large from SRAG notebook)
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "mxbai-embed-large"

    # ------------------------------------------------------------------
    # Qdrant (vector store — same as BasicRAG project)
    # ------------------------------------------------------------------
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "csrag_documents"

    # Embedding dimension for mxbai-embed-large
    embedding_dimension: int = 1024

    # ------------------------------------------------------------------
    # PostgreSQL (Long-Term Memory store)
    # ------------------------------------------------------------------
    postgres_uri: str = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

    # ------------------------------------------------------------------
    # Tavily (web search — used by CRAG on non-CORRECT retrieval)
    # ------------------------------------------------------------------
    tavily_api_key: str
    tavily_max_results: int = 5

    # ------------------------------------------------------------------
    # Document processing (chunk settings — from SRAG notebook)
    # ------------------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_k: int = 4

    # ------------------------------------------------------------------
    # Short-Term Memory (STM)
    # ------------------------------------------------------------------
    stm_message_threshold: int = 6  # summarize when messages exceed this

    # ------------------------------------------------------------------
    # CRAG thresholds
    # ------------------------------------------------------------------
    crag_upper_threshold: float = 0.7   # >= this => CORRECT
    crag_lower_threshold: float = 0.3   # all below this => INCORRECT, else AMBIGUOUS

    # ------------------------------------------------------------------
    # SRAG loop limits
    # ------------------------------------------------------------------
    srag_max_retries: int = 2       # max revise-answer iterations
    max_rewrite_tries: int = 2      # max question-rewrite + re-retrieve iterations

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    allowed_origins: str = "*"

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ------------------------------------------------------------------
    # Application info
    # ------------------------------------------------------------------
    app_name: str = "CSRAG — Corrective Self-Reflective RAG"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance."""
    return Settings()

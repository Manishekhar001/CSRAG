from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str

    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    memory_llm_model: str = "llama-3.3-70b-versatile"
    memory_llm_temperature: float = 0.0

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "mxbai-embed-large"

    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "csrag_documents"

    embedding_dimension: int = 1024

    postgres_uri: str = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

    tavily_api_key: str
    tavily_max_results: int = 5

    chunk_size: int = 900
    chunk_overlap: int = 150

    retrieval_k: int = 4

    stm_message_threshold: int = 6

    crag_upper_threshold: float = 0.7
    crag_lower_threshold: float = 0.3

    srag_max_retries: int = 2
    max_rewrite_tries: int = 2

    log_level: str = "INFO"

    allowed_origins: str = "*"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    app_name: str = "CSRAG — Corrective Self-Reflective RAG"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()

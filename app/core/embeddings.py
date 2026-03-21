from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    logger.info(
        f"Initialising Ollama embeddings: model={settings.embedding_model}, "
        f"base_url={settings.ollama_base_url}"
    )
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    logger.info("Ollama embeddings initialised successfully")
    return embeddings


class EmbeddingsService:
    def __init__(self) -> None:
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model = settings.embedding_model

    def embed_query(self, text: str) -> list[float]:
        logger.debug(f"Embedding query: {text[:60]}...")
        return self.embeddings.embed_query(text)

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        logger.debug(f"Embedding {len(docs)} documents")
        return self.embeddings.embed_documents(docs)

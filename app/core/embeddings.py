"""Embeddings module — OllamaEmbeddings (mxbai-embed-large).

Uses the same embedding model as the SRAG/CRAG notebooks (local Ollama).
The Qdrant collection is configured with embedding_dimension=1024 to match
mxbai-embed-large's output size.
"""

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    """Return a cached OllamaEmbeddings instance.

    Returns:
        Configured :class:`OllamaEmbeddings` instance pointing at the
        locally running Ollama server.
    """
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
    """Thin service wrapper around OllamaEmbeddings."""

    def __init__(self) -> None:
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model = settings.embedding_model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text.

        Returns:
            Embedding vector as a list of floats.
        """
        logger.debug(f"Embedding query: {text[:60]}...")
        return self.embeddings.embed_query(text)

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """Embed a batch of document strings.

        Args:
            docs: List of document strings.

        Returns:
            List of embedding vectors.
        """
        logger.debug(f"Embedding {len(docs)} documents")
        return self.embeddings.embed_documents(docs)

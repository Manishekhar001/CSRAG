from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    logger.info("Qdrant client connected successfully")
    return client


class VectorStoreService:
    def __init__(self, collection_name: str | None = None) -> None:
        self.client = get_qdrant_client()
        self.collection_name = collection_name or settings.collection_name
        self.embeddings = get_embeddings()
        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        logger.info(f"VectorStoreService ready — collection: {self.collection_name}")

    def _ensure_collection(self) -> None:
        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}' exists "
                f"({info.points_count} points)"
            )
        except Exception as e:
            err_str = str(e).lower()
            is_not_found = (
                isinstance(e, UnexpectedResponse)
                or "not found" in err_str
                or "doesn't exist" in err_str
                or "does not exist" in err_str
            )
            if is_not_found:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created")
            else:
                raise

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            logger.warning("add_documents called with empty list")
            return []

        ids = [str(uuid4()) for _ in documents]
        logger.info(f"Adding {len(documents)} chunks to Qdrant")
        self.vector_store.add_documents(documents=documents, ids=ids)
        logger.info(f"Successfully added {len(documents)} chunks")
        return ids

    def search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.retrieval_k
        if not query:
            logger.warning("search called with empty query")
            return []
        logger.debug(f"Searching Qdrant: '{query[:60]}...' (k={k})")
        results = self.vector_store.similarity_search(query=query, k=k)
        logger.debug(f"Found {len(results)} results")
        return results

    def search_with_score(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = k or settings.retrieval_k
        if not query:
            return []
        logger.debug(f"Scored search: '{query[:60]}...' (k={k})")
        return self.vector_store.similarity_search_with_score(query=query, k=k)

    def get_retriever(self, k: int | None = None) -> Any:
        k = k or settings.retrieval_k
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def delete_collection(self) -> None:
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")

    def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }
        except UnexpectedResponse:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "indexed_vectors_count": 0,
                "status": "not_found",
            }

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

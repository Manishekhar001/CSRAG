"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test-qdrant-key"
os.environ["TAVILY_API_KEY"] = "test-tavily-key"
os.environ["POSTGRES_URI"] = "postgresql://postgres:postgres@localhost:5432/test"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("app.config.get_settings") as mock:
        settings = MagicMock()
        settings.groq_api_key = "test-groq-key"
        settings.qdrant_url = "http://localhost:6333"
        settings.qdrant_api_key = "test-qdrant-key"
        settings.tavily_api_key = "test-tavily-key"
        settings.collection_name = "csrag_documents"
        settings.chunk_size = 900
        settings.chunk_overlap = 150
        settings.embedding_model = "mxbai-embed-large"
        settings.embedding_dimension = 1024
        settings.llm_model = "llama-3.3-70b-versatile"
        settings.llm_temperature = 0.0
        settings.memory_llm_model = "llama-3.3-70b-versatile"
        settings.memory_llm_temperature = 0.0
        settings.retrieval_k = 4
        settings.stm_message_threshold = 6
        settings.crag_upper_threshold = 0.7
        settings.crag_lower_threshold = 0.3
        settings.srag_max_retries = 2
        settings.max_rewrite_tries = 2
        settings.tavily_max_results = 5
        settings.log_level = "WARNING"
        settings.api_host = "0.0.0.0"
        settings.api_port = 8000
        settings.app_name = "CSRAG — Corrective Self-Reflective RAG"
        settings.app_version = "0.1.0"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    with patch("app.core.vector_store.get_qdrant_client") as mock:
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        client.get_collection.return_value = MagicMock(
            points_count=10,
            indexed_vectors_count=10,
            status=MagicMock(value="green"),
        )
        mock.return_value = client
        yield client


@pytest.fixture
def mock_embeddings():
    """Mock Ollama embeddings."""
    with patch("app.core.embeddings.get_embeddings") as mock:
        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1] * 1024
        embeddings.embed_documents.return_value = [[0.1] * 1024]
        mock.return_value = embeddings
        yield embeddings


@pytest.fixture
def mock_vector_store(mock_qdrant_client, mock_embeddings):
    """Mock vector store service."""
    with patch("app.main.VectorStoreService") as mock:
        service = MagicMock()
        service.health_check.return_value = True
        service.get_collection_info.return_value = {
            "name": "csrag_documents",
            "points_count": 10,
            "indexed_vectors_count": 10,
            "status": "green",
        }
        service.add_documents.return_value = ["id-1", "id-2"]
        service.search.return_value = []
        service.search_with_score.return_value = []
        service.delete_collection.return_value = None
        mock.return_value = service
        yield service


@pytest.fixture
def mock_engine():
    """Mock CSRAGEngine with working async query and stream methods."""
    with patch("app.main.CSRAGEngine") as mock_cls:
        engine = MagicMock()
        engine.health_check.return_value = True

        async def _aquery(question, thread_id, user_id):
            return {
                "answer": "This is a test answer.",
                "sources": [
                    {
                        "content": "Test content from document.",
                        "metadata": {"source": "test.pdf"},
                        "origin": "internal",
                    }
                ],
                "crag_verdict": "CORRECT",
                "crag_reason": "At least one chunk scored >= 0.7 (max=0.85)",
                "issup": "fully_supported",
                "evidence": ["Supporting quote from context."],
                "isuse": "useful",
                "use_reason": "Answer directly addresses the question.",
                "retries": 0,
                "rewrite_tries": 0,
            }

        async def _astream(question, thread_id, user_id):
            yield "This is "
            yield "a streamed "
            yield "answer."

        engine.aquery = _aquery
        engine.astream = _astream
        mock_cls.return_value = engine
        yield engine


@pytest.fixture
def mock_postgres_store():
    """Mock PostgresStore with correct context manager pattern."""
    with patch("app.main.PostgresStore") as mock_cls:
        store_cm = MagicMock()
        store = MagicMock()
        store.setup.return_value = None
        store.search.return_value = []
        store_cm.__enter__ = MagicMock(return_value=store)
        store_cm.__exit__ = MagicMock(return_value=False)
        mock_cls.from_conn_string.return_value = store_cm
        yield store


@pytest.fixture
def mock_postgres_saver():
    """Mock PostgresSaver with correct context manager pattern."""
    with patch("app.main.PostgresSaver") as mock_cls:
        checkpointer_cm = MagicMock()
        checkpointer = MagicMock()
        checkpointer.setup.return_value = None
        checkpointer_cm.__enter__ = MagicMock(return_value=checkpointer)
        checkpointer_cm.__exit__ = MagicMock(return_value=False)
        mock_cls.from_conn_string.return_value = checkpointer_cm
        yield checkpointer


@pytest.fixture
def client(mock_vector_store, mock_engine, mock_postgres_store, mock_postgres_saver):
    """Create test client with all dependencies mocked."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_vector_store_unhealthy(mock_qdrant_client, mock_embeddings):
    """Mock VectorStoreService where Qdrant health check fails."""
    with patch("app.main.VectorStoreService") as mock_cls:
        svc = MagicMock()
        svc.health_check.return_value = False
        svc.get_collection_info.return_value = {
            "name": "csrag_documents",
            "points_count": 0,
            "indexed_vectors_count": 0,
            "status": "not_found",
        }
        mock_cls.return_value = svc
        yield svc


@pytest.fixture
def client_qdrant_down(
    mock_vector_store_unhealthy, mock_engine, mock_postgres_store, mock_postgres_saver
):
    """Create test client where Qdrant is unreachable."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return (
        "This is a sample document for testing the CSRAG pipeline.\n\n"
        "Section 1: Introduction\n"
        "CSRAG combines Corrective RAG, Self-Reflective RAG, and Memory.\n\n"
        "Section 2: Components\n"
        "- CRAG evaluator\n"
        "- SRAG verifier\n"
        "- Short-term memory\n"
        "- Long-term memory\n"
    )


@pytest.fixture
def sample_text_bytes(sample_text_content):
    """Sample text file bytes for upload tests."""
    return sample_text_content.encode("utf-8")


@pytest.fixture
def sample_csv_bytes():
    """Sample CSV file bytes for upload tests."""
    return b"name,age,city\nAlice,30,New York\nBob,25,London\n"


@pytest.fixture
def sample_chunks():
    """Sample document chunks for processor tests."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="This is chunk 1 about CRAG.",
            metadata={"source": "test.txt", "chunk": 0},
        ),
        Document(
            page_content="This is chunk 2 about SRAG.",
            metadata={"source": "test.txt", "chunk": 1},
        ),
    ]

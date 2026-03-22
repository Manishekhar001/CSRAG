import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test-qdrant-key"
os.environ["TAVILY_API_KEY"] = "test-tavily-key"
os.environ["POSTGRES_URI"] = "postgresql://postgres:postgres@localhost:5432/test"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture
def mock_vector_store():
    with patch("app.main.VectorStoreService") as mock_cls:
        svc = MagicMock()
        svc.health_check.return_value = True
        svc.get_collection_info.return_value = {
            "name": "csrag_documents",
            "points_count": 10,
            "indexed_vectors_count": 10,
            "status": "green",
        }
        svc.add_documents.return_value = ["id-1", "id-2"]
        svc.search.return_value = []
        svc.search_with_score.return_value = []
        svc.delete_collection.return_value = None
        mock_cls.return_value = svc
        yield svc


@pytest.fixture
def mock_engine():
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
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_vector_store_unhealthy():
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
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_text_bytes():
    return b"This is a sample text document for testing the CSRAG pipeline."


@pytest.fixture
def sample_csv_bytes():
    return b"name,age,city\nAlice,30,New York\nBob,25,London\n"

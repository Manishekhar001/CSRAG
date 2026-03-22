"""Tests for chat endpoints."""


class TestChatEndpoints:
    """Test chat API endpoints."""

    def test_chat_endpoint(self, client, mock_engine):
        """Test basic chat endpoint."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "processing_time_ms" in data
        assert data["question"] == "What is the refund policy?"

    def test_chat_without_sources(self, client, mock_engine):
        """Test chat without sources."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": False,
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] is None

    def test_chat_with_sources(self, client, mock_engine):
        """Test chat with sources included."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] is not None
        assert isinstance(data["sources"], list)

    def test_chat_sources_have_origin_field(self, client, mock_engine):
        """Test that each source document has an origin field."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        data = response.json()
        for source in data["sources"]:
            assert source["origin"] in ("internal", "web")

    def test_chat_returns_crag_verdict(self, client, mock_engine):
        """Test that CRAG verdict is returned in the response."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        data = response.json()
        assert "crag_verdict" in data
        assert data["crag_verdict"] == "CORRECT"

    def test_chat_returns_srag_fields(self, client, mock_engine):
        """Test that SRAG support and usefulness fields are returned."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        data = response.json()
        assert data["issup"] == "fully_supported"
        assert data["isuse"] == "useful"
        assert isinstance(data["evidence"], list)

    def test_chat_returns_loop_counters(self, client, mock_engine):
        """Test that retry and rewrite counters are present in the response."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        data = response.json()
        assert data["retries"] == 0
        assert data["rewrite_tries"] == 0

    def test_chat_calls_engine_aquery(self, client, mock_engine):
        """Test that the chat endpoint calls the engine with correct arguments."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        client.post("/chat", json=request_data)

        mock_engine.aquery.assert_called_once_with(
            question="What is the refund policy?",
            thread_id="thread-test-001",
            user_id="user-test-001",
        )

    def test_chat_stream_endpoint(self, client, mock_engine):
        """Test streaming chat endpoint returns 200."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat/stream", json=request_data)

        assert response.status_code == 200

    def test_chat_stream_returns_text_plain(self, client, mock_engine):
        """Test streaming endpoint returns text/plain content type."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat/stream", json=request_data)

        assert "text/plain" in response.headers["content-type"]

    def test_chat_stream_returns_content(self, client, mock_engine):
        """Test streaming endpoint returns non-empty content."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat/stream", json=request_data)

        assert len(response.content) > 0


class TestChatValidation:
    """Test chat request validation."""

    def test_chat_empty_question(self, client):
        """Test chat with empty question returns 422."""
        request_data = {
            "question": "",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
            "include_sources": True,
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 422

    def test_chat_missing_question(self, client):
        """Test chat without question field returns 422."""
        request_data = {
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 422

    def test_chat_missing_thread_id(self, client):
        """Test chat without thread_id returns 422."""
        request_data = {
            "question": "What is the refund policy?",
            "user_id": "user-test-001",
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 422

    def test_chat_missing_user_id(self, client):
        """Test chat without user_id returns 422."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 422

    def test_chat_question_max_length(self, client):
        """Test chat question max length validation."""
        request_data = {
            "question": "a" * 2001,
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 422

    def test_chat_valid_question_length(self, client, mock_engine):
        """Test valid question length passes validation."""
        request_data = {
            "question": "What is the refund policy?",
            "thread_id": "thread-test-001",
            "user_id": "user-test-001",
        }

        response = client.post("/chat", json=request_data)

        assert response.status_code == 200

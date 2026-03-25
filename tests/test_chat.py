"""Tests for chat endpoints."""

from unittest.mock import AsyncMock, MagicMock


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


class TestChatHistory:
    """Test chat history endpoint."""

    def _make_checkpoint_tuple(self, messages: list, summary: str = "") -> MagicMock:
        """Build a mock CheckpointTuple with the given messages and summary."""
        from langchain_core.messages import AIMessage, HumanMessage

        msg_objs = []
        for msg in messages:
            if msg["role"] == "human":
                msg_objs.append(HumanMessage(content=msg["content"]))
            else:
                msg_objs.append(AIMessage(content=msg["content"]))

        checkpoint = {"channel_values": {"messages": msg_objs, "summary": summary}}
        tup = MagicMock()
        tup.checkpoint = checkpoint
        return tup

    def test_history_returns_200_for_existing_thread(self, client, mock_postgres_saver):
        """Test that history returns 200 when the thread exists."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([
                {"role": "human", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ])
        )

        response = client.get("/chat/history/thread-test-001")

        assert response.status_code == 200

    def test_history_returns_correct_thread_id(self, client, mock_postgres_saver):
        """Test that history response includes the correct thread_id."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([])
        )

        response = client.get("/chat/history/thread-test-001")

        assert response.json()["thread_id"] == "thread-test-001"

    def test_history_returns_messages_in_order(self, client, mock_postgres_saver):
        """Test that messages are returned in chronological order."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([
                {"role": "human", "content": "What is RAG?"},
                {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
                {"role": "human", "content": "How does it work?"},
                {"role": "assistant", "content": "It retrieves relevant documents first."},
            ])
        )

        response = client.get("/chat/history/thread-test-001")
        data = response.json()

        assert data["message_count"] == 4
        assert data["messages"][0]["role"] == "human"
        assert data["messages"][0]["content"] == "What is RAG?"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][2]["role"] == "human"
        assert data["messages"][3]["role"] == "assistant"

    def test_history_returns_correct_roles(self, client, mock_postgres_saver):
        """Test that message roles are correctly mapped to human/assistant."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([
                {"role": "human", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ])
        )

        response = client.get("/chat/history/thread-test-001")
        messages = response.json()["messages"]

        assert messages[0]["role"] == "human"
        assert messages[1]["role"] == "assistant"

    def test_history_returns_summary_when_present(self, client, mock_postgres_saver):
        """Test that the rolling STM summary is included when available."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple(
                messages=[
                    {"role": "human", "content": "Latest question"},
                    {"role": "assistant", "content": "Latest answer"},
                ],
                summary="User asked about RAG and embeddings in earlier turns.",
            )
        )

        response = client.get("/chat/history/thread-test-001")
        data = response.json()

        assert data["summary"] == "User asked about RAG and embeddings in earlier turns."

    def test_history_returns_empty_summary_when_none(self, client, mock_postgres_saver):
        """Test that summary is empty string when no compression has happened."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([
                {"role": "human", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ])
        )

        response = client.get("/chat/history/thread-test-001")

        assert response.json()["summary"] == ""

    def test_history_returns_404_for_unknown_thread(self, client, mock_postgres_saver):
        """Test that 404 is returned for a thread_id that has never been used."""
        mock_postgres_saver.aget_tuple = AsyncMock(return_value=None)

        response = client.get("/chat/history/thread-does-not-exist")

        assert response.status_code == 404

    def test_history_returns_empty_message_list_for_new_thread(self, client, mock_postgres_saver):
        """Test that an existing but empty thread returns an empty messages list."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([])
        )

        response = client.get("/chat/history/thread-test-001")
        data = response.json()

        assert data["message_count"] == 0
        assert data["messages"] == []

    def test_history_message_count_matches_messages_length(self, client, mock_postgres_saver):
        """Test that message_count matches the actual length of the messages list."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_checkpoint_tuple([
                {"role": "human", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "human", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ])
        )

        response = client.get("/chat/history/thread-test-001")
        data = response.json()

        assert data["message_count"] == len(data["messages"])

    def _make_serialized_checkpoint_tuple(self, messages: list, summary: str = "") -> MagicMock:
        """Build a mock CheckpointTuple with serialized dict messages (as returned by actual checkpointer)."""
        msg_dicts = []
        for msg in messages:
            if msg["role"] == "human":
                msg_dicts.append({"type": "human", "content": msg["content"]})
            else:
                msg_dicts.append({"type": "ai", "content": msg["content"]})

        checkpoint = {"channel_values": {"messages": msg_dicts, "summary": summary}}
        tup = MagicMock()
        tup.checkpoint = checkpoint
        return tup

    def test_history_handles_serialized_messages(self, client, mock_postgres_saver):
        """Test that history correctly handles serialized dict messages from checkpointer."""
        mock_postgres_saver.aget_tuple = AsyncMock(
            return_value=self._make_serialized_checkpoint_tuple([
                {"role": "human", "content": "Hello from dict"},
                {"role": "assistant", "content": "Hi there from dict!"},
            ])
        )

        response = client.get("/chat/history/thread-test-001")
        data = response.json()

        assert response.status_code == 200
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "human"
        assert data["messages"][0]["content"] == "Hello from dict"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Hi there from dict!"


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

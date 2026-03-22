"""Tests for memory services and endpoints."""

from unittest.mock import MagicMock, patch

import pytest


class TestSTMSummarizer:
    """Test suite for STMSummarizer class."""

    @pytest.fixture
    def stm(self):
        """Create STMSummarizer with mocked LLM."""
        with patch("app.core.memory.stm.ChatGroq") as mock_cls:
            llm = MagicMock()
            response = MagicMock()
            response.content = "This is a summary of the conversation."
            llm.invoke.return_value = response
            mock_cls.return_value = llm
            with patch("app.core.memory.stm.get_settings") as mock_settings:
                settings = MagicMock()
                settings.llm_model = "llama-3.3-70b-versatile"
                settings.llm_temperature = 0.0
                settings.groq_api_key = "test-key"
                settings.stm_message_threshold = 6
                mock_settings.return_value = settings
                from app.core.memory.stm import STMSummarizer
                return STMSummarizer()

    def _make_message(self, content):
        """Helper to create a mock message."""
        msg = MagicMock()
        msg.content = content
        msg.id = f"msg-{hash(content)}"
        return msg

    def test_should_summarize_returns_false_below_threshold(self, stm):
        """Test that summarize is not triggered below the message threshold."""
        messages = [self._make_message(f"msg {i}") for i in range(5)]
        assert stm.should_summarize(messages) is False

    def test_should_summarize_returns_true_above_threshold(self, stm):
        """Test that summarize is triggered above the message threshold."""
        messages = [self._make_message(f"msg {i}") for i in range(7)]
        assert stm.should_summarize(messages) is True

    def test_should_summarize_returns_false_at_threshold(self, stm):
        """Test that summarize is not triggered at exactly the threshold."""
        messages = [self._make_message(f"msg {i}") for i in range(6)]
        assert stm.should_summarize(messages) is False

    def test_summarize_returns_new_summary_string(self, stm):
        """Test that summarize returns a non-empty summary string."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = stm.summarize(messages, "")
        assert isinstance(new_summary, str)
        assert len(new_summary) > 0

    def test_summarize_keeps_last_two_messages(self, stm):
        """Test that summarize marks all but the last two messages for deletion."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = stm.summarize(messages, "")
        assert len(remove_ops) == len(messages) - 2

    def test_summarize_extends_existing_summary(self, stm):
        """Test that summarize incorporates an existing summary."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, _ = stm.summarize(messages, "Existing summary text.")
        assert isinstance(new_summary, str)
        stm._llm.invoke.assert_called_once()
        call_args = stm._llm.invoke.call_args[0][0]
        assert any("Existing summary" in str(m.content) for m in call_args)

    def test_summarize_without_existing_summary(self, stm):
        """Test that summarize works correctly with no prior summary."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = stm.summarize(messages, "")
        assert isinstance(new_summary, str)


class TestLTMService:
    """Test suite for LTMService class."""

    @pytest.fixture
    def ltm(self):
        """Create LTMService with mocked LLM."""
        with patch("app.core.memory.ltm.ChatGroq") as mock_cls:
            llm = MagicMock()
            mock_cls.return_value = llm
            with patch("app.core.memory.ltm.get_settings") as mock_settings:
                settings = MagicMock()
                settings.memory_llm_model = "llama-3.3-70b-versatile"
                settings.memory_llm_temperature = 0.0
                settings.groq_api_key = "test-key"
                mock_settings.return_value = settings
                from app.core.memory.ltm import LTMService
                return LTMService()

    def test_namespace_returns_correct_tuple(self, ltm):
        """Test that namespace builds the correct tuple for a user."""
        ns = ltm._namespace("user-123")
        assert ns == ("user", "user-123", "details")

    def test_read_memories_returns_empty_when_no_items(self, ltm):
        """Test read_memories returns '(empty)' when the store has no items."""
        store = MagicMock()
        store.search.return_value = []
        result = ltm.read_memories(store, "user-123")
        assert result == "(empty)"

    def test_read_memories_joins_items(self, ltm):
        """Test read_memories joins all stored facts into a single string."""
        store = MagicMock()
        item1 = MagicMock()
        item1.value = {"data": "User is Nitish."}
        item2 = MagicMock()
        item2.value = {"data": "User teaches AI."}
        store.search.return_value = [item1, item2]

        result = ltm.read_memories(store, "user-123")
        assert "User is Nitish." in result
        assert "User teaches AI." in result

    def test_extract_and_store_writes_new_facts(self, ltm):
        """Test that new facts are written to the store."""
        store = MagicMock()
        store.search.return_value = []

        decision = MagicMock()
        decision.should_write = True
        memory = MagicMock()
        memory.is_new = True
        memory.text = "User's name is Nitish."
        decision.memories = [memory]
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.return_value = decision

        written = ltm.extract_and_store(store, "user-123", "Hi, I am Nitish.")
        assert written == 1
        store.put.assert_called_once()

    def test_extract_and_store_skips_duplicate_facts(self, ltm):
        """Test that duplicate facts (is_new=False) are not written."""
        store = MagicMock()
        store.search.return_value = []

        decision = MagicMock()
        decision.should_write = True
        memory = MagicMock()
        memory.is_new = False
        memory.text = "User is Nitish."
        decision.memories = [memory]
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.return_value = decision

        written = ltm.extract_and_store(store, "user-123", "I am Nitish.")
        assert written == 0
        store.put.assert_not_called()

    def test_extract_and_store_returns_zero_when_nothing_to_write(self, ltm):
        """Test that zero is returned when there is nothing memory-worthy."""
        store = MagicMock()
        store.search.return_value = []

        decision = MagicMock()
        decision.should_write = False
        decision.memories = []
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.return_value = decision

        written = ltm.extract_and_store(store, "user-123", "Hello!")
        assert written == 0

    def test_extract_and_store_returns_zero_on_llm_error(self, ltm):
        """Test graceful error handling when LLM extraction fails."""
        store = MagicMock()
        store.search.return_value = []
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.side_effect = Exception("LLM error")

        written = ltm.extract_and_store(store, "user-123", "Hello!")
        assert written == 0


class TestMemoryEndpoints:
    """Test memory API endpoints."""

    def test_list_memories(self, client):
        """Test listing memories for a user."""
        response = client.get("/memory/user-test-001")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-test-001"
        assert "memories" in data
        assert "count" in data

    def test_list_memories_empty(self, client):
        """Test listing memories returns empty list when none stored."""
        response = client.get("/memory/user-test-001")

        data = response.json()
        assert data["count"] == 0
        assert isinstance(data["memories"], list)

    def test_list_memories_count_matches_list_length(self, client):
        """Test that count field matches the length of the memories list."""
        response = client.get("/memory/user-test-001")

        data = response.json()
        assert data["count"] == len(data["memories"])

    def test_list_memories_with_stored_data(self, client, mock_postgres_store):
        """Test listing memories returns stored facts correctly."""
        item = MagicMock()
        item.value = {"data": "User's name is Nitish."}
        item.key = "some-uuid"
        mock_postgres_store.search.return_value = [item]

        response = client.get("/memory/user-test-001")
        data = response.json()
        assert data["count"] == 1
        assert data["memories"][0]["data"] == "User's name is Nitish."

    def test_list_memories_different_user_ids(self, client):
        """Test that different user IDs return separate memory scopes."""
        response_1 = client.get("/memory/user-001")
        response_2 = client.get("/memory/user-002")

        assert response_1.status_code == 200
        assert response_2.status_code == 200
        assert response_1.json()["user_id"] == "user-001"
        assert response_2.json()["user_id"] == "user-002"

    def test_delete_memories(self, client, mock_postgres_store):
        """Test deleting all memories for a user."""
        response = client.delete("/memory/user-test-001")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-test-001"
        assert "message" in data

    def test_delete_memories_calls_store_delete(self, client, mock_postgres_store):
        """Test that delete endpoint calls store.delete for each item."""
        item = MagicMock()
        item.value = {"data": "Some fact."}
        item.key = "uuid-to-delete"
        mock_postgres_store.search.return_value = [item]

        client.delete("/memory/user-test-001")

        mock_postgres_store.delete.assert_called_once()

    def test_delete_memories_when_empty(self, client, mock_postgres_store):
        """Test deleting memories when none exist returns success."""
        mock_postgres_store.search.return_value = []
        response = client.delete("/memory/user-test-001")

        assert response.status_code == 200
        assert "Deleted 0" in response.json()["message"]

"""Tests for memory services and endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

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
            llm.ainvoke = AsyncMock(return_value=response)
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

    async def test_summarize_returns_new_summary_string(self, stm):
        """Test that summarize returns a non-empty summary string."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = await stm.summarize(messages, "")
        assert isinstance(new_summary, str)
        assert len(new_summary) > 0

    async def test_summarize_keeps_last_two_messages(self, stm):
        """Test that summarize marks all but the last two messages for deletion."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = await stm.summarize(messages, "")
        assert len(remove_ops) == len(messages) - 2

    async def test_summarize_extends_existing_summary(self, stm):
        """Test that summarize incorporates an existing summary."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, _ = await stm.summarize(messages, "Existing summary text.")
        assert isinstance(new_summary, str)
        stm._llm.ainvoke.assert_called_once()
        call_args = stm._llm.ainvoke.call_args[0][0]
        assert any("Existing summary" in str(m.content) for m in call_args)

    async def test_summarize_without_existing_summary(self, stm):
        """Test that summarize works correctly with no prior summary."""
        messages = [self._make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = await stm.summarize(messages, "")
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

    def _make_store(self, search_result=None):
        """Helper to create an async-compatible mock store."""
        store = MagicMock()
        store.asearch = AsyncMock(return_value=search_result or [])
        store.aput = AsyncMock(return_value=None)
        store.adelete = AsyncMock(return_value=None)
        return store

    def test_namespace_returns_correct_tuple(self, ltm):
        """Test that namespace builds the correct tuple for a user."""
        ns = ltm._namespace("user-123")
        assert ns == ("user", "user-123", "details")

    async def test_read_memories_returns_empty_when_no_items(self, ltm):
        """Test read_memories returns '(empty)' when the store has no items."""
        store = self._make_store(search_result=[])
        result = await ltm.read_memories(store, "user-123")
        assert result == "(empty)"

    async def test_read_memories_joins_items(self, ltm):
        """Test read_memories joins all stored facts into a single string."""
        item1 = MagicMock()
        item1.value = {"data": "User is Nitish."}
        item2 = MagicMock()
        item2.value = {"data": "User teaches AI."}
        store = self._make_store(search_result=[item1, item2])

        result = await ltm.read_memories(store, "user-123")
        assert "User is Nitish." in result
        assert "User teaches AI." in result

    async def test_extract_and_store_writes_new_facts(self, ltm):
        """Test that new facts are written to the store."""
        store = self._make_store(search_result=[])

        decision = MagicMock()
        decision.should_write = True
        memory = MagicMock()
        memory.is_new = True
        memory.text = "User's name is Nitish."
        decision.memories = [memory]
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.return_value = decision

        written = await ltm.extract_and_store(store, "user-123", "Hi, I am Nitish.")
        assert written == 1
        store.aput.assert_called_once()

    async def test_extract_and_store_skips_duplicate_facts(self, ltm):
        """Test that duplicate facts (is_new=False) are not written."""
        store = self._make_store(search_result=[])

        decision = MagicMock()
        decision.should_write = True
        memory = MagicMock()
        memory.is_new = False
        memory.text = "User is Nitish."
        decision.memories = [memory]
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.return_value = decision

        written = await ltm.extract_and_store(store, "user-123", "I am Nitish.")
        assert written == 0
        store.aput.assert_not_called()

    async def test_extract_and_store_returns_zero_when_nothing_to_write(self, ltm):
        """Test that zero is returned when there is nothing memory-worthy."""
        store = self._make_store(search_result=[])

        decision = MagicMock()
        decision.should_write = False
        decision.memories = []
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.return_value = decision

        written = await ltm.extract_and_store(store, "user-123", "Hello!")
        assert written == 0

    async def test_extract_and_store_returns_zero_on_llm_error(self, ltm):
        """Test graceful error handling when LLM extraction fails."""
        store = self._make_store(search_result=[])
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.side_effect = Exception("LLM error")

        written = await ltm.extract_and_store(store, "user-123", "Hello!")
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
        mock_postgres_store.asearch.return_value = [item]

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
        """Test that delete endpoint calls store.adelete for each item."""
        item = MagicMock()
        item.value = {"data": "Some fact."}
        item.key = "uuid-to-delete"
        mock_postgres_store.asearch.return_value = [item]

        client.delete("/memory/user-test-001")

        mock_postgres_store.adelete.assert_called_once()

    def test_delete_memories_when_empty(self, client, mock_postgres_store):
        """Test deleting memories when none exist returns success."""
        mock_postgres_store.asearch.return_value = []
        response = client.delete("/memory/user-test-001")

        assert response.status_code == 200
        assert "Deleted 0" in response.json()["message"]

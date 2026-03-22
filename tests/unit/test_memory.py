from unittest.mock import MagicMock, patch

import pytest

from app.core.memory.stm import STMSummarizer
from app.core.memory.ltm import LTMService


@pytest.fixture
def stm():
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
            return STMSummarizer()


@pytest.fixture
def ltm():
    with patch("app.core.memory.ltm.ChatGroq") as mock_cls:
        llm = MagicMock()
        mock_cls.return_value = llm
        with patch("app.core.memory.ltm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.memory_llm_model = "llama-3.3-70b-versatile"
            settings.memory_llm_temperature = 0.0
            settings.groq_api_key = "test-key"
            mock_settings.return_value = settings
            return LTMService()


def _make_message(content, role="human"):
    msg = MagicMock()
    msg.content = content
    msg.id = f"msg-{hash(content)}"
    return msg


class TestSTMSummarizer:
    def test_should_summarize_returns_false_below_threshold(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(5)]
        assert stm.should_summarize(messages) is False

    def test_should_summarize_returns_true_above_threshold(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(7)]
        assert stm.should_summarize(messages) is True

    def test_should_summarize_returns_false_at_threshold(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(6)]
        assert stm.should_summarize(messages) is False

    def test_summarize_returns_new_summary_string(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = stm.summarize(messages, "")
        assert isinstance(new_summary, str)
        assert len(new_summary) > 0

    def test_summarize_keeps_last_two_messages(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = stm.summarize(messages, "")
        assert len(remove_ops) == len(messages) - 2

    def test_summarize_with_existing_summary_extends_it(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(8)]
        new_summary, _ = stm.summarize(messages, "Existing summary text.")
        assert isinstance(new_summary, str)
        stm._llm.invoke.assert_called_once()
        call_args = stm._llm.invoke.call_args[0][0]
        assert any("Existing summary" in str(m.content) for m in call_args)

    def test_summarize_without_existing_summary(self, stm):
        messages = [_make_message(f"msg {i}") for i in range(8)]
        new_summary, remove_ops = stm.summarize(messages, "")
        assert isinstance(new_summary, str)


class TestLTMService:
    def test_namespace_returns_correct_tuple(self, ltm):
        ns = ltm._namespace("user-123")
        assert ns == ("user", "user-123", "details")

    def test_read_memories_returns_empty_when_no_items(self, ltm):
        store = MagicMock()
        store.search.return_value = []
        result = ltm.read_memories(store, "user-123")
        assert result == "(empty)"

    def test_read_memories_joins_items(self, ltm):
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
        store = MagicMock()
        store.search.return_value = []
        ltm._extractor = MagicMock()
        ltm._extractor.invoke.side_effect = Exception("LLM error")

        written = ltm.extract_and_store(store, "user-123", "Hello!")
        assert written == 0

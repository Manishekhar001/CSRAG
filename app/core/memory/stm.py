"""Short-Term Memory (STM) via conversation summarisation.

When the message count exceeds settings.stm_message_threshold (default 6),
the summarise node:
  1. Asks the LLM to summarise the conversation (extending any existing summary).
  2. Deletes all messages except the last 2 via RemoveMessage.
  3. Stores the new summary in state['summary'].

The summary is prepended as a system message in the generate_answer node so the
LLM always has prior context without an ever-growing message window.

This is a direct productionisation of 5_stm_summarization.ipynb.
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class STMSummarizer:
    """Summarises long conversations to keep the message window compact."""

    def __init__(self) -> None:
        settings = get_settings()
        self._threshold = settings.stm_message_threshold
        self._llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
        )
        logger.info(
            f"STMSummarizer ready — threshold={self._threshold} messages"
        )

    def should_summarize(self, messages: list) -> bool:
        """Return True if the message count exceeds the threshold.

        Args:
            messages: Current message list from state.

        Returns:
            True if summarisation should run.
        """
        return len(messages) > self._threshold

    def summarize(self, messages: list, existing_summary: str) -> tuple[str, list]:
        """Generate a new summary and return messages to delete.

        Args:
            messages: Full current message list.
            existing_summary: Any previously generated summary string.

        Returns:
            Tuple of (new_summary_string, list_of_RemoveMessage_objects).
        """
        if existing_summary:
            prompt_text = (
                f"Existing summary:\n{existing_summary}\n\n"
                "Extend the summary to include the new conversation above. "
                "Be concise."
            )
        else:
            prompt_text = (
                "Summarise the conversation above concisely. "
                "Capture key facts, user preferences, and conclusions."
            )

        messages_for_summary = list(messages) + [
            HumanMessage(content=prompt_text)
        ]

        logger.info(
            f"Summarising conversation — {len(messages)} messages, "
            f"existing_summary={'yes' if existing_summary else 'no'}"
        )

        response = self._llm.invoke(messages_for_summary)
        new_summary: str = response.content

        # Keep only the last 2 messages; delete the rest
        messages_to_delete = messages[:-2]
        remove_ops = [RemoveMessage(id=m.id) for m in messages_to_delete]

        logger.info(
            f"Summary generated — deleted {len(remove_ops)} messages, "
            f"kept {len(messages) - len(remove_ops)}"
        )
        return new_summary, remove_ops


@lru_cache
def get_stm_summarizer() -> STMSummarizer:
    """Return a cached :class:`STMSummarizer` instance."""
    return STMSummarizer()

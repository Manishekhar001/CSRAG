"""Long-Term Memory (LTM) via PostgresStore.

Uses LangGraph's PostgresStore to persist user-specific facts across
sessions permanently.  Namespace pattern: ("user", user_id, "details").

The remember node:
  1. Reads existing memories from Postgres.
  2. Passes them + the latest user message to a structured LLM extractor.
  3. Writes only *new* atomic facts back to Postgres (deduplication via is_new flag).

This is a direct productionisation of 8_ltm_postgres.ipynb.
"""

import uuid
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Pydantic schemas for structured LTM extraction
# ------------------------------------------------------------------

class MemoryItem(BaseModel):
    """A single atomic memory extracted from a user message."""

    text: str = Field(..., description="Short atomic fact about the user.")
    is_new: bool = Field(
        ...,
        description="True if this fact is NEW vs existing memories, False if duplicate.",
    )


class MemoryDecision(BaseModel):
    """LLM decision on whether to write memories for this message."""

    should_write: bool = Field(
        ...,
        description="True if there is any memory-worthy information in the message.",
    )
    memories: list[MemoryItem] = Field(default_factory=list)


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

_MEMORY_PROMPT = """\
You are responsible for maintaining accurate long-term user memory.

CURRENT USER DETAILS (existing memories):
{existing_memories}

TASK:
- Review the user's latest message.
- Extract user-specific information worth storing long-term:
    identity, stable preferences, ongoing projects, goals, tools used.
- For each item set is_new=true ONLY if it adds genuinely NEW information
  compared to CURRENT USER DETAILS.
- If it duplicates existing memory, set is_new=false.
- Keep each memory as one short atomic sentence.
- No speculation — only facts explicitly stated by the user.
- If nothing is memory-worthy, return should_write=false and an empty list.
"""


# ------------------------------------------------------------------
# Service class
# ------------------------------------------------------------------

class LTMService:
    """Manages long-term memory read/write operations via PostgresStore."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGroq(
            model=settings.memory_llm_model,
            temperature=settings.memory_llm_temperature,
            api_key=settings.groq_api_key,
        )
        self._extractor = self._llm.with_structured_output(MemoryDecision)
        logger.info(
            f"LTMService ready — "
            f"model={settings.memory_llm_model}"
        )

    @staticmethod
    def _namespace(user_id: str) -> tuple:
        """Return the Postgres namespace tuple for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            Namespace tuple ``("user", user_id, "details")``.
        """
        return ("user", user_id, "details")

    def read_memories(self, store, user_id: str) -> str:
        """Read all stored facts for a user and return them as a string.

        Args:
            store: LangGraph BaseStore instance (PostgresStore).
            user_id: Unique user identifier.

        Returns:
            Newline-joined facts string, or "(empty)" if none exist.
        """
        ns = self._namespace(user_id)
        items = store.search(ns)
        if not items:
            logger.debug(f"LTM: no memories found for user={user_id}")
            return "(empty)"
        memories = "\n".join(it.value.get("data", "") for it in items)
        logger.debug(
            f"LTM: read {len(items)} memories for user={user_id}"
        )
        return memories

    def extract_and_store(
        self, store, user_id: str, user_message: str
    ) -> int:
        """Extract new facts from a user message and persist them.

        Args:
            store: LangGraph BaseStore instance (PostgresStore).
            user_id: Unique user identifier.
            user_message: The latest message text from the user.

        Returns:
            Number of new facts written.
        """
        existing = self.read_memories(store, user_id)
        ns = self._namespace(user_id)

        try:
            decision: MemoryDecision = self._extractor.invoke(
                [
                    SystemMessage(
                        content=_MEMORY_PROMPT.format(existing_memories=existing)
                    ),
                    {"role": "user", "content": user_message},
                ]
            )
        except Exception as e:
            logger.error(f"LTM extraction failed: {e}")
            return 0

        written = 0
        if decision.should_write:
            for mem in decision.memories:
                if mem.is_new and mem.text.strip():
                    store.put(
                        ns,
                        str(uuid.uuid4()),
                        {"data": mem.text.strip()},
                    )
                    written += 1
                    logger.debug(f"LTM stored: '{mem.text.strip()}'")

        logger.info(
            f"LTM extraction done — {written} new facts stored for user={user_id}"
        )
        return written


@lru_cache
def get_ltm_service() -> LTMService:
    """Return a cached :class:`LTMService` instance."""
    return LTMService()

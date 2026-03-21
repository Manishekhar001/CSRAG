"""Graph state definition for the CSRAG pipeline.

A single TypedDict that carries every field across all nodes.
Using Annotated[list[BaseMessage], add_messages] for the messages field
gives us the LangGraph reducer — new messages are appended, not overwritten.
"""

from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CSRAGState(TypedDict):
    """Complete state carried through the CSRAG graph."""

    # ------------------------------------------------------------------
    # Conversation (STM backbone — add_messages reducer)
    # ------------------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]

    # Short-term memory rolling summary (may be empty string)
    summary: str

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------
    # User identifier — passed via RunnableConfig["configurable"]["user_id"]
    user_id: str

    # Formatted LTM facts injected into the system prompt
    ltm_context: str

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    # Whether the question requires document retrieval
    need_retrieval: bool

    # ------------------------------------------------------------------
    # Retrieval & CRAG
    # ------------------------------------------------------------------
    # The raw question text (extracted from messages[-1])
    question: str

    # Rewritten retrieval query (may differ from question after CRAG rewrite)
    retrieval_query: str

    # Number of times we have rewritten the question and re-retrieved
    rewrite_tries: int

    # Raw retrieved documents from Qdrant
    docs: list[Document]

    # Documents that passed the CRAG relevance filter (score > LOWER_TH)
    good_docs: list[Document]

    # CRAG verdict for the current retrieval batch
    crag_verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT", ""]

    # Human-readable reason for the verdict
    crag_reason: str

    # Rewritten web-search query (CRAG non-CORRECT path)
    web_query: str

    # Documents fetched from Tavily web search
    web_docs: list[Document]

    # ------------------------------------------------------------------
    # Context refinement
    # ------------------------------------------------------------------
    # All sentence strips decomposed from the raw context
    strips: list[str]

    # Strips kept after the LLM keep/drop filter
    kept_strips: list[str]

    # Final refined context string passed to generate_answer
    refined_context: str

    # ------------------------------------------------------------------
    # Generation & SRAG
    # ------------------------------------------------------------------
    # The generated answer text
    answer: str

    # SRAG support verdict
    issup: Literal["fully_supported", "partially_supported", "no_support", ""]

    # Evidence quotes extracted by SRAG support verifier
    evidence: list[str]

    # Number of revise-answer attempts so far
    retries: int

    # SRAG usefulness verdict
    isuse: Literal["useful", "not_useful", ""]

    # Reason given by the SRAG usefulness judge
    use_reason: str

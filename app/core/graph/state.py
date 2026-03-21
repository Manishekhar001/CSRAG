from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CSRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    user_id: str
    ltm_context: str
    need_retrieval: bool
    question: str
    retrieval_query: str
    rewrite_tries: int
    docs: list[Document]
    good_docs: list[Document]
    crag_verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT", ""]
    crag_reason: str
    web_query: str
    web_docs: list[Document]
    strips: list[str]
    kept_strips: list[str]
    refined_context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support", ""]
    evidence: list[str]
    retries: int
    isuse: Literal["useful", "not_useful", ""]
    use_reason: str

"""CRAG document evaluator.

Scores each retrieved chunk 0–1 and classifies the batch as:
    CORRECT   — at least one chunk scored >= UPPER_TH (0.7)
    INCORRECT — every chunk scored < LOWER_TH (0.3)
    AMBIGUOUS — anything in between

This is a direct productionisation of the scoring logic in 6_ambiguous.ipynb.
"""

from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ------------------------------------------------------------------

class DocEvalScore(BaseModel):
    """LLM-graded relevance score for a single retrieved chunk."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score in [0.0, 1.0]. 1.0 = chunk alone answers the question fully.",
    )
    reason: str = Field(..., description="Short justification for the score.")


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

_DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for a RAG system.\n"
            "You will be given ONE retrieved document chunk and a user question.\n"
            "Return a relevance score in [0.0, 1.0].\n\n"
            "Scoring guide:\n"
            "  1.0 — chunk alone is sufficient to fully answer the question\n"
            "  0.7 — chunk contains strong, directly relevant information\n"
            "  0.5 — chunk is partially relevant (related topic, incomplete answer)\n"
            "  0.3 — chunk is marginally relevant (same domain, no direct answer)\n"
            "  0.0 — chunk is completely irrelevant\n\n"
            "Be conservative with high scores. Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)


# ------------------------------------------------------------------
# Verdict type
# ------------------------------------------------------------------

CRAGVerdict = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]


# ------------------------------------------------------------------
# Service class
# ------------------------------------------------------------------

class CRAGEvaluator:
    """Evaluates retrieved document chunks and produces a CRAG verdict."""

    def __init__(self) -> None:
        settings = get_settings()
        self._upper_th = settings.crag_upper_threshold
        self._lower_th = settings.crag_lower_threshold

        llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
        )
        self._eval_chain = _DOC_EVAL_PROMPT | llm.with_structured_output(DocEvalScore)

        logger.info(
            f"CRAGEvaluator ready — "
            f"upper_th={self._upper_th}, lower_th={self._lower_th}"
        )

    def evaluate(
        self, question: str, docs: list[Document]
    ) -> tuple[CRAGVerdict, str, list[Document]]:
        """Score each document and return the batch verdict.

        Args:
            question: The user's question.
            docs: Retrieved documents to evaluate.

        Returns:
            Tuple of (verdict, reason, good_docs) where ``good_docs`` contains
            all documents that scored above ``lower_th``.
        """
        if not docs:
            logger.warning("CRAGEvaluator.evaluate called with empty docs list")
            return "INCORRECT", "No documents retrieved", []

        scores: list[float] = []
        good_docs: list[Document] = []

        for doc in docs:
            try:
                result: DocEvalScore = self._eval_chain.invoke(
                    {"question": question, "chunk": doc.page_content}
                )
                scores.append(result.score)
                logger.debug(
                    f"Chunk scored {result.score:.2f} — {result.reason[:80]}"
                )
                if result.score > self._lower_th:
                    good_docs.append(doc)
            except Exception as e:
                logger.error(f"Doc eval failed for chunk: {e}")
                scores.append(0.0)

        # Classify verdict
        if any(s >= self._upper_th for s in scores):
            verdict: CRAGVerdict = "CORRECT"
            reason = (
                f"At least one chunk scored >= {self._upper_th} "
                f"(max={max(scores):.2f})"
            )
        elif all(s < self._lower_th for s in scores):
            verdict = "INCORRECT"
            reason = (
                f"All chunks scored < {self._lower_th} "
                f"(max={max(scores):.2f})"
            )
            good_docs = []
        else:
            verdict = "AMBIGUOUS"
            reason = (
                f"No chunk >= {self._upper_th} but not all < {self._lower_th} "
                f"(max={max(scores):.2f})"
            )

        logger.info(f"CRAG verdict: {verdict} — {reason}")
        return verdict, reason, good_docs


@lru_cache
def get_crag_evaluator() -> CRAGEvaluator:
    """Return a cached :class:`CRAGEvaluator` instance."""
    return CRAGEvaluator()

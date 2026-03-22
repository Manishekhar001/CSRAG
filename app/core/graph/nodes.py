import asyncio
import json
import re
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.crag.evaluator import get_crag_evaluator
from app.core.crag.web_search import get_web_search_service
from app.core.graph.state import CSRAGState
from app.core.memory.ltm import get_ltm_service
from app.core.memory.stm import get_stm_summarizer
from app.core.srag.verifier import get_srag_verifier
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache
def _get_chat_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )


def _build_system_prompt(ltm_context: str, summary: str) -> str:
    base = (
        "You are a knowledgeable and helpful assistant with memory capabilities.\n\n"
        "Answer questions clearly and concisely using the provided context.\n"
        "If no context is available, use your general knowledge.\n"
        "If you don't know the answer, say so clearly.\n"
        "Do not make up information."
    )

    sections: list[str] = []

    if ltm_context and ltm_context != "(empty)":
        sections.append(f"Long-term user memory:\n{ltm_context}")

    if summary:
        sections.append(f"Recent conversation summary:\n{summary}")

    if sections:
        return base + "\n\n" + "\n\n".join(sections)

    return base


# ---------------------------------------------------------------------------
# LTM
# ---------------------------------------------------------------------------

async def ltm_remember_node(
    state: CSRAGState,
    config: RunnableConfig,
    *,
    store,
) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "default")
    ltm = get_ltm_service()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    user_message = last_human.content if last_human else ""

    await ltm.extract_and_store(store, user_id, user_message)
    ltm_context = await ltm.read_memories(store, user_id)

    logger.info(f"LTM remember done for user={user_id}")
    return {"user_id": user_id, "ltm_context": ltm_context}


# ---------------------------------------------------------------------------
# Retrieval decision
# ---------------------------------------------------------------------------

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description=(
            "True if answering requires specific facts from ingested documents. "
            "False for general knowledge questions."
        ),
    )


_DECIDE_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether document retrieval is needed to answer the question.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "  True  — question requires specific facts from company/domain documents.\n"
            "  False — question is general knowledge or a simple definition.\n"
            "  When unsure, choose True.",
        ),
        ("human", "Question: {question}"),
    ]
)


async def decide_retrieval_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async LLM call."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    question = last_human.content if last_human else ""

    llm = _get_chat_llm()
    decider = _DECIDE_RETRIEVAL_PROMPT | llm.with_structured_output(RetrieveDecision)

    try:
        decision: RetrieveDecision = await decider.ainvoke({"question": question})
        need_retrieval = decision.should_retrieve
    except Exception as e:
        logger.error(f"decide_retrieval failed: {e} — defaulting to True")
        need_retrieval = True

    logger.info(f"Decide retrieval: need_retrieval={need_retrieval}, q='{question[:80]}'")
    return {
        "question": question,
        "need_retrieval": need_retrieval,
        "retrieval_query": question,
    }


# ---------------------------------------------------------------------------
# Direct generation (no retrieval path)
# ---------------------------------------------------------------------------

async def generate_direct_node(state: CSRAGState) -> dict:
    """
    Bug 4 fix: async LLM call.
    Bug 8 fix: SRAG fields set to 'skipped' — SRAG did NOT run on this path,
               so we must not lie and claim full support/usefulness.
    """
    llm = _get_chat_llm()
    system_msg = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )
    messages = [SystemMessage(content=system_msg)] + list(state["messages"])
    response = await llm.ainvoke(messages)
    answer = response.content
    logger.info("generate_direct completed")
    return {
        "answer": answer,
        # Bug 8 fix: honest labels — SRAG was not run on this path
        "issup": "skipped",
        "isuse": "skipped",
        "evidence": [],
        "use_reason": "direct generation — SRAG not applicable without retrieval context",
    }


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------

async def retrieve_docs_node(state: CSRAGState, *, vector_store: VectorStoreService) -> dict:
    """Bug 4 fix: blocking Qdrant call offloaded to thread pool."""
    query = state.get("retrieval_query") or state["question"]
    logger.info(f"Retrieving docs for: '{query[:80]}'")
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None, lambda: vector_store.search(query, k=settings.retrieval_k)
    )
    logger.info(f"Retrieved {len(docs)} docs")
    return {"docs": docs}


# ---------------------------------------------------------------------------
# CRAG evaluation
# ---------------------------------------------------------------------------

async def evaluate_docs_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async — calls the now-async CRAGEvaluator."""
    evaluator = get_crag_evaluator()
    verdict, reason, good_docs = await evaluator.evaluate(
        question=state["question"],
        docs=state.get("docs", []),
    )
    return {
        "crag_verdict": verdict,
        "crag_reason": reason,
        "good_docs": good_docs,
    }


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

async def rewrite_query_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async."""
    svc = get_web_search_service()
    web_query = await svc.rewrite_query(state["question"])
    return {"web_query": web_query}


async def web_search_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async."""
    svc = get_web_search_service()
    query = state.get("web_query") or state["question"]
    web_docs = await svc.search(query)
    return {"web_docs": web_docs}


# ---------------------------------------------------------------------------
# Context refinement — BATCH sentence scoring (Bug 3 fix)
# ---------------------------------------------------------------------------

def _decompose_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class BatchFilterResult(BaseModel):
    kept_indices: list[int] = Field(
        ...,
        description=(
            "0-based indices of sentences (from the provided list) that directly "
            "help answer the question. Return an empty list if none are relevant."
        ),
    )


_BATCH_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter for a RAG system.\n"
            "Given a question and a numbered list of sentences, return the 0-based "
            "indices of ONLY the sentences that directly and specifically help answer "
            "the question.\n"
            "Discard sentences that are tangential, generic, or off-topic.\n"
            "Output JSON only with key: kept_indices (list of integers).",
        ),
        (
            "human",
            "Question: {question}\n\nSentences (JSON array):\n{sentences_json}",
        ),
    ]
)


async def refine_context_node(state: CSRAGState) -> dict:
    """
    Bug 3 fix: replaced N serial LLM calls (one per sentence) with a single
               batched call that scores all sentences at once.
    Bug 4 fix: fully async.
    """
    verdict = state.get("crag_verdict", "CORRECT")
    good_docs = state.get("good_docs", [])
    web_docs = state.get("web_docs", [])

    if verdict == "CORRECT":
        docs_to_use = good_docs
    elif verdict == "INCORRECT":
        docs_to_use = web_docs
    else:
        docs_to_use = good_docs + web_docs

    raw_context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    if not raw_context:
        logger.warning("refine_context: empty context — skipping sentence filter")
        return {"strips": [], "kept_strips": [], "refined_context": ""}

    strips = _decompose_to_sentences(raw_context)

    if not strips:
        return {"strips": [], "kept_strips": [], "refined_context": raw_context}

    llm = _get_chat_llm()
    filter_chain = _BATCH_FILTER_PROMPT | llm.with_structured_output(BatchFilterResult)

    kept: list[str] = strips  # safe default — keep everything on failure
    try:
        result: BatchFilterResult = await filter_chain.ainvoke(
            {
                "question": state["question"],
                "sentences_json": json.dumps(strips),
            }
        )
        valid_indices = {i for i in result.kept_indices if 0 <= i < len(strips)}
        kept = [strips[i] for i in sorted(valid_indices)]
    except Exception as e:
        logger.error(f"Batch sentence filter failed: {e} — keeping all strips")

    refined_context = "\n".join(kept)
    logger.info(
        f"refine_context: {len(strips)} strips → {len(kept)} kept "
        f"(verdict={verdict})"
    )
    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}\n\n"
            "Answer the question using ONLY the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't have enough "
            "information to answer that based on the available documents.'\n"
            "Do not make up information.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


async def generate_answer_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async LLM call."""
    llm = _get_chat_llm()
    system_prompt = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )
    response = await (_RAG_PROMPT | llm).ainvoke(
        {
            "system_prompt": system_prompt,
            "context": state.get("refined_context", ""),
            "question": state["question"],
        }
    )
    answer = response.content
    logger.info("generate_answer completed")
    return {"answer": answer}


# ---------------------------------------------------------------------------
# SRAG verification & revision
# ---------------------------------------------------------------------------

async def verify_support_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async."""
    verifier = get_srag_verifier()
    verdict, evidence = await verifier.verify_support(
        question=state["question"],
        context=state.get("refined_context", ""),
        answer=state["answer"],
    )
    return {"issup": verdict, "evidence": evidence}


async def revise_answer_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async."""
    verifier = get_srag_verifier()
    revised = await verifier.revise_answer(
        question=state["question"],
        context=state.get("refined_context", ""),
        answer=state["answer"],
    )
    new_retries = state.get("retries", 0) + 1
    logger.info(f"revise_answer: attempt {new_retries}")
    return {"answer": revised, "retries": new_retries}


async def verify_usefulness_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async."""
    verifier = get_srag_verifier()
    verdict, reason = await verifier.verify_usefulness(
        question=state["question"],
        answer=state["answer"],
    )
    return {"isuse": verdict, "use_reason": reason}


# ---------------------------------------------------------------------------
# Question rewrite
# ---------------------------------------------------------------------------

class RewrittenQuestion(BaseModel):
    query: str = Field(
        ...,
        description="Rewritten retrieval query to get better documents.",
    )


_REWRITE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The previous answer was not useful. Reformulate the question into a "
            "better document retrieval query.\n"
            "Rules:\n"
            "  - Keep it specific and concrete (8–15 words).\n"
            "  - Focus on what information would actually answer the question.\n"
            "  - Return JSON with key: query.",
        ),
        ("human", "Original question: {question}"),
    ]
)


async def rewrite_question_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async LLM call."""
    llm = _get_chat_llm()
    chain = _REWRITE_QUESTION_PROMPT | llm.with_structured_output(RewrittenQuestion)
    try:
        result: RewrittenQuestion = await chain.ainvoke({"question": state["question"]})
        new_query = result.query
    except Exception as e:
        logger.error(f"rewrite_question failed: {e} — using original question")
        new_query = state["question"]

    new_tries = state.get("rewrite_tries", 0) + 1
    logger.info(f"rewrite_question: '{new_query}' (attempt {new_tries})")
    return {"retrieval_query": new_query, "rewrite_tries": new_tries}


# ---------------------------------------------------------------------------
# STM summarization
# ---------------------------------------------------------------------------

async def stm_summarize_node(state: CSRAGState) -> dict:
    """Bug 4 fix: async summarize call."""
    answer = state.get("answer", "")
    ai_msg = AIMessage(content=answer)

    summarizer = get_stm_summarizer()
    all_messages = list(state["messages"]) + [ai_msg]

    if summarizer.should_summarize(all_messages):
        logger.info("STM threshold exceeded — summarising conversation")
        new_summary, remove_ops = await summarizer.summarize(
            messages=all_messages,
            existing_summary=state.get("summary", ""),
        )
        return {"messages": [ai_msg] + remove_ops, "summary": new_summary}

    return {"messages": [ai_msg]}


# ---------------------------------------------------------------------------
# Routing functions (pure logic — stay sync, no LLM calls)
# ---------------------------------------------------------------------------

def route_after_decide(
    state: CSRAGState,
) -> Literal["generate_direct", "retrieve_docs"]:
    return "retrieve_docs" if state["need_retrieval"] else "generate_direct"


def route_after_crag(
    state: CSRAGState,
) -> Literal["refine_context", "rewrite_query"]:
    return "refine_context" if state["crag_verdict"] == "CORRECT" else "rewrite_query"


def route_after_support(
    state: CSRAGState,
) -> Literal["revise_answer", "verify_usefulness"]:
    issup = state.get("issup", "fully_supported")
    retries = state.get("retries", 0)
    if issup != "fully_supported" and retries < settings.srag_max_retries:
        return "revise_answer"
    return "verify_usefulness"


def route_after_usefulness(
    state: CSRAGState,
) -> Literal["rewrite_question", "stm_summarize"]:
    isuse = state.get("isuse", "useful")
    rewrite_tries = state.get("rewrite_tries", 0)
    if isuse == "not_useful" and rewrite_tries < settings.max_rewrite_tries:
        return "rewrite_question"
    return "stm_summarize"

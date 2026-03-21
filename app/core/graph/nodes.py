"""All LangGraph graph nodes for the CSRAG pipeline.

Each node is a pure function:
    (state: CSRAGState, ...) -> dict   # partial state update

Nodes that need the Postgres store receive it as a keyword-only argument
injected by LangGraph when the graph is compiled with store=<PostgresStore>.

Node order in the graph:
  START
    → ltm_remember
    → decide_retrieval
        → [need_retrieval=False] generate_direct → stm_summarize → END
        → [need_retrieval=True]  retrieve_docs
            → evaluate_docs (CRAG)
                → [CORRECT]            refine_context
                → [AMBIGUOUS/INCORRECT] rewrite_query → web_search → refine_context
            → refine_context
            → generate_answer
            → verify_support (SRAG)
                → [not fully_supported + retries < max] revise_answer → verify_support (loop)
                → [fully_supported or max retries]     verify_usefulness
                    → [not_useful + rewrite_tries < max] rewrite_question → retrieve_docs (loop)
                    → [useful or max retries]            stm_summarize → END
"""

import re
from typing import Literal

from langchain_core.documents import Document
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


# ======================================================================
# Shared LLM (chat)
# ======================================================================

def _get_chat_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )


# ======================================================================
# System prompt template (injected with STM summary + LTM facts)
# ======================================================================

_SYSTEM_PROMPT_TEMPLATE = """\
You are a knowledgeable and helpful assistant with memory capabilities.

{ltm_section}

{stm_section}

Answer questions clearly and concisely using the provided context.
If no context is available, use your general knowledge.
If you don't know the answer, say so clearly.
Do not make up information.
"""


def _build_system_prompt(ltm_context: str, summary: str) -> str:
    ltm_section = (
        f"Long-term user memory:\n{ltm_context}"
        if ltm_context and ltm_context != "(empty)"
        else ""
    )
    stm_section = (
        f"Recent conversation summary:\n{summary}"
        if summary
        else ""
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        ltm_section=ltm_section,
        stm_section=stm_section,
    ).strip()


# ======================================================================
# Node 1: LTM remember
# ======================================================================

def ltm_remember_node(
    state: CSRAGState,
    config: RunnableConfig,
    *,
    store,
) -> dict:
    """Extract and persist new facts from the latest user message.

    Also reads all existing memories and stores them in state['ltm_context']
    so downstream nodes can inject them without hitting Postgres again.

    Args:
        state: Current graph state.
        config: LangGraph runtime config (contains user_id).
        store: PostgresStore injected by LangGraph.

    Returns:
        Partial state update with user_id and ltm_context.
    """
    user_id = config.get("configurable", {}).get("user_id", "default")
    ltm = get_ltm_service()

    # Latest user message text
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    user_message = last_human.content if last_human else ""

    # Extract + store new facts
    ltm.extract_and_store(store, user_id, user_message)

    # Read back all facts for downstream use
    ltm_context = ltm.read_memories(store, user_id)

    logger.info(f"LTM remember done for user={user_id}")
    return {"user_id": user_id, "ltm_context": ltm_context}


# ======================================================================
# Node 2: Decide retrieval
# ======================================================================

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


def decide_retrieval_node(state: CSRAGState) -> dict:
    """Decide whether to retrieve documents or answer directly.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with question, need_retrieval, retrieval_query.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    question = last_human.content if last_human else ""

    llm = _get_chat_llm()
    decider = _DECIDE_RETRIEVAL_PROMPT | llm.with_structured_output(RetrieveDecision)

    try:
        decision: RetrieveDecision = decider.invoke({"question": question})
        need_retrieval = decision.should_retrieve
    except Exception as e:
        logger.error(f"decide_retrieval failed: {e} — defaulting to True")
        need_retrieval = True

    logger.info(f"Decide retrieval: need_retrieval={need_retrieval}, q='{question[:80]}'")
    return {
        "question": question,
        "need_retrieval": need_retrieval,
        "retrieval_query": question,  # default; may be overwritten by rewrite nodes
        "rewrite_tries": state.get("rewrite_tries", 0),
        "retries": state.get("retries", 0),
    }


# ======================================================================
# Node 3: Generate direct (no retrieval)
# ======================================================================

_DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using your general knowledge.\n"
            "If the question requires specific company or domain information, say: "
            "'I don't have enough context to answer that specifically.'\n"
            "Do not make up information.",
        ),
        ("human", "{question}"),
    ]
)


def generate_direct_node(state: CSRAGState) -> dict:
    """Generate an answer without retrieval (general knowledge path).

    Args:
        state: Current graph state.

    Returns:
        Partial state update with answer and issup/isuse set to accepted values.
    """
    llm = _get_chat_llm()
    system_msg = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )

    messages = [SystemMessage(content=system_msg)] + list(state["messages"])
    response = llm.invoke(messages)
    answer = response.content

    logger.info("generate_direct completed")
    return {
        "answer": answer,
        "issup": "fully_supported",
        "isuse": "useful",
        "evidence": [],
        "use_reason": "direct generation — no retrieval",
    }


# ======================================================================
# Node 4: Retrieve docs
# ======================================================================

def retrieve_docs_node(state: CSRAGState, *, vector_store: VectorStoreService) -> dict:
    """Retrieve top-k documents from Qdrant.

    Args:
        state: Current graph state.
        vector_store: Injected VectorStoreService instance.

    Returns:
        Partial state update with docs.
    """
    query = state.get("retrieval_query") or state["question"]
    logger.info(f"Retrieving docs for: '{query[:80]}'")
    docs = vector_store.search(query, k=settings.retrieval_k)
    logger.info(f"Retrieved {len(docs)} docs")
    return {"docs": docs}


# ======================================================================
# Node 5: Evaluate docs (CRAG)
# ======================================================================

def evaluate_docs_node(state: CSRAGState) -> dict:
    """Score each retrieved doc and produce a CRAG verdict.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with crag_verdict, crag_reason, good_docs.
    """
    evaluator = get_crag_evaluator()
    verdict, reason, good_docs = evaluator.evaluate(
        question=state["question"],
        docs=state.get("docs", []),
    )
    return {
        "crag_verdict": verdict,
        "crag_reason": reason,
        "good_docs": good_docs,
    }


# ======================================================================
# Node 6: Rewrite query (CRAG — non-CORRECT path)
# ======================================================================

def rewrite_query_node(state: CSRAGState) -> dict:
    """Rewrite the user question into an optimised web-search query.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with web_query.
    """
    svc = get_web_search_service()
    web_query = svc.rewrite_query(state["question"])
    return {"web_query": web_query}


# ======================================================================
# Node 7: Web search (CRAG)
# ======================================================================

def web_search_node(state: CSRAGState) -> dict:
    """Execute a Tavily web search using the rewritten query.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with web_docs.
    """
    svc = get_web_search_service()
    query = state.get("web_query") or state["question"]
    web_docs = svc.search(query)
    return {"web_docs": web_docs}


# ======================================================================
# Node 8: Refine context (sentence-level filter)
# ======================================================================

def _decompose_to_sentences(text: str) -> list[str]:
    """Split text into atomic sentences.

    Args:
        text: Raw context string.

    Returns:
        List of non-trivial sentences (> 20 chars).
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool = Field(
        ...,
        description="True if the sentence directly helps answer the question.",
    )


_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true ONLY if the sentence directly and specifically helps "
            "answer the question.\n"
            "Use ONLY the sentence provided. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)


def refine_context_node(state: CSRAGState) -> dict:
    """Sentence-level context refinement.

    Merges internal and web docs based on CRAG verdict, then filters at
    sentence level using an LLM keep/drop judge.

    Verdict → docs used:
      CORRECT   → good_docs only
      INCORRECT → web_docs only
      AMBIGUOUS → good_docs + web_docs

    Args:
        state: Current graph state.

    Returns:
        Partial state update with strips, kept_strips, refined_context.
    """
    verdict = state.get("crag_verdict", "CORRECT")
    good_docs = state.get("good_docs", [])
    web_docs = state.get("web_docs", [])

    if verdict == "CORRECT":
        docs_to_use = good_docs
    elif verdict == "INCORRECT":
        docs_to_use = web_docs
    else:  # AMBIGUOUS
        docs_to_use = good_docs + web_docs

    raw_context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    if not raw_context:
        logger.warning("refine_context: empty context — skipping sentence filter")
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }

    strips = _decompose_to_sentences(raw_context)

    llm = _get_chat_llm()
    filter_chain = _FILTER_PROMPT | llm.with_structured_output(KeepOrDrop)

    kept: list[str] = []
    for sentence in strips:
        try:
            result: KeepOrDrop = filter_chain.invoke(
                {"question": state["question"], "sentence": sentence}
            )
            if result.keep:
                kept.append(sentence)
        except Exception as e:
            logger.error(f"Sentence filter error: {e}")
            kept.append(sentence)  # keep on error to avoid empty context

    refined_context = "\n".join(kept)
    logger.info(
        f"refine_context: {len(strips)} strips → {len(kept)} kept "
        f"(verdict={verdict})"
    )
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }


# ======================================================================
# Node 9: Generate answer
# ======================================================================

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


def generate_answer_node(state: CSRAGState) -> dict:
    """Generate the final answer from the refined context.

    Injects both the STM summary and LTM facts into the system prompt.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with answer.
    """
    llm = _get_chat_llm()
    system_prompt = _build_system_prompt(
        state.get("ltm_context", ""),
        state.get("summary", ""),
    )

    response = (_RAG_PROMPT | llm).invoke(
        {
            "system_prompt": system_prompt,
            "context": state.get("refined_context", ""),
            "question": state["question"],
        }
    )
    answer = response.content
    logger.info("generate_answer completed")
    return {"answer": answer}


# ======================================================================
# Node 10: Verify support (SRAG)
# ======================================================================

def verify_support_node(state: CSRAGState) -> dict:
    """Check factual grounding of the answer.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with issup and evidence.
    """
    verifier = get_srag_verifier()
    verdict, evidence = verifier.verify_support(
        question=state["question"],
        context=state.get("refined_context", ""),
        answer=state["answer"],
    )
    return {"issup": verdict, "evidence": evidence}


# ======================================================================
# Node 11: Revise answer (SRAG)
# ======================================================================

def revise_answer_node(state: CSRAGState) -> dict:
    """Rewrite the answer to remove unsupported claims.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with revised answer and incremented retries.
    """
    verifier = get_srag_verifier()
    revised = verifier.revise_answer(
        question=state["question"],
        context=state.get("refined_context", ""),
        answer=state["answer"],
    )
    new_retries = state.get("retries", 0) + 1
    logger.info(f"revise_answer: attempt {new_retries}")
    return {"answer": revised, "retries": new_retries}


# ======================================================================
# Node 12: Verify usefulness (SRAG)
# ======================================================================

def verify_usefulness_node(state: CSRAGState) -> dict:
    """Check whether the answer actually helps the user.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with isuse and use_reason.
    """
    verifier = get_srag_verifier()
    verdict, reason = verifier.verify_usefulness(
        question=state["question"],
        answer=state["answer"],
    )
    return {"isuse": verdict, "use_reason": reason}


# ======================================================================
# Node 13: Rewrite question (SRAG — not_useful path)
# ======================================================================

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


def rewrite_question_node(state: CSRAGState) -> dict:
    """Reformulate the question for a better retrieval attempt.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with retrieval_query and incremented rewrite_tries.
    """
    llm = _get_chat_llm()
    chain = _REWRITE_QUESTION_PROMPT | llm.with_structured_output(RewrittenQuestion)
    try:
        result: RewrittenQuestion = chain.invoke({"question": state["question"]})
        new_query = result.query
    except Exception as e:
        logger.error(f"rewrite_question failed: {e} — using original question")
        new_query = state["question"]

    new_tries = state.get("rewrite_tries", 0) + 1
    logger.info(f"rewrite_question: '{new_query}' (attempt {new_tries})")
    return {"retrieval_query": new_query, "rewrite_tries": new_tries}


# ======================================================================
# Node 14: STM summarize
# ======================================================================

def stm_summarize_node(state: CSRAGState) -> dict:
    """Append the assistant answer to messages, then optionally summarise.

    Appends the generated answer as an AIMessage, then checks if the total
    message count exceeds the STM threshold. If so, summarises and prunes.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with messages (including new AIMessage + any
        RemoveMessage ops) and possibly an updated summary.
    """
    answer = state.get("answer", "")
    ai_msg = AIMessage(content=answer)

    summarizer = get_stm_summarizer()
    all_messages = list(state["messages"]) + [ai_msg]

    if summarizer.should_summarize(all_messages):
        logger.info("STM threshold exceeded — summarising conversation")
        new_summary, remove_ops = summarizer.summarize(
            messages=all_messages,
            existing_summary=state.get("summary", ""),
        )
        return {
            "messages": [ai_msg] + remove_ops,
            "summary": new_summary,
        }

    return {"messages": [ai_msg]}


# ======================================================================
# Routing functions (conditional edges)
# ======================================================================

def route_after_decide(
    state: CSRAGState,
) -> Literal["generate_direct", "retrieve_docs"]:
    """Route after decide_retrieval node.

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    return "retrieve_docs" if state["need_retrieval"] else "generate_direct"


def route_after_crag(
    state: CSRAGState,
) -> Literal["refine_context", "rewrite_query"]:
    """Route after evaluate_docs node.

    Args:
        state: Current graph state.

    Returns:
        'refine_context' if CORRECT, 'rewrite_query' otherwise.
    """
    return "refine_context" if state["crag_verdict"] == "CORRECT" else "rewrite_query"


def route_after_support(
    state: CSRAGState,
) -> Literal["revise_answer", "verify_usefulness"]:
    """Route after verify_support node.

    Args:
        state: Current graph state.

    Returns:
        'revise_answer' if not fully supported and retries remain,
        'verify_usefulness' otherwise.
    """
    issup = state.get("issup", "fully_supported")
    retries = state.get("retries", 0)

    if issup != "fully_supported" and retries < settings.srag_max_retries:
        return "revise_answer"
    return "verify_usefulness"


def route_after_usefulness(
    state: CSRAGState,
) -> Literal["rewrite_question", "stm_summarize"]:
    """Route after verify_usefulness node.

    Args:
        state: Current graph state.

    Returns:
        'rewrite_question' if not useful and rewrite budget remains,
        'stm_summarize' otherwise (accept the answer).
    """
    isuse = state.get("isuse", "useful")
    rewrite_tries = state.get("rewrite_tries", 0)

    if isuse == "not_useful" and rewrite_tries < settings.max_rewrite_tries:
        return "rewrite_question"
    return "stm_summarize"

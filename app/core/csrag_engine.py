from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.config import get_settings
from app.core.graph.builder import build_graph
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_RECURSION_LIMIT = 80


class CSRAGEngine:
    def __init__(
        self,
        vector_store: VectorStoreService,
        store: AsyncPostgresStore,
        checkpointer: AsyncPostgresSaver,
    ) -> None:
        self._graph = build_graph(
            vector_store=vector_store,
            store=store,
            checkpointer=checkpointer,
        )
        logger.info("CSRAGEngine initialised")

    def _build_config(self, thread_id: str, user_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": _RECURSION_LIMIT,
        }

    @staticmethod
    def _initial_state(question: str) -> dict:
        return {
            "messages": [HumanMessage(content=question)],
            "summary": "",
            "user_id": "",
            "ltm_context": "",
            "need_retrieval": False,
            "question": "",
            "retrieval_query": "",
            "rewrite_tries": 0,
            "docs": [],
            "good_docs": [],
            "crag_verdict": "",
            "crag_reason": "",
            "web_query": "",
            "web_docs": [],
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "answer": "",
            "issup": "",
            "evidence": [],
            "retries": 0,
            "isuse": "",
            "use_reason": "",
        }

    # Bug 5 fix: removed the sync query() method entirely.
    # It was unsafe because ltm_remember_node is async — calling graph.invoke()
    # (sync) with an async node causes a deadlock inside a running event loop.
    # All callers should use aquery() instead.

    async def aquery(self, question: str, thread_id: str, user_id: str) -> dict:
        logger.info(
            f"async query — thread={thread_id}, user={user_id}, "
            f"q='{question[:80]}'"
        )
        config = self._build_config(thread_id, user_id)
        result = await self._graph.ainvoke(self._initial_state(question), config)
        return self._format_result(result)

    async def astream(self, question: str, thread_id: str, user_id: str):
        """
        Bug 7 fix: switched from stream_mode='values' (which yielded full answer
        blobs on each state change) to stream_mode='messages' which yields actual
        message tokens/chunks as the LLM generates them.

        We filter to only stream chunks from the answer-generation nodes so that
        intermediate LLM calls (decide_retrieval, CRAG eval, SRAG, etc.) are not
        leaked to the client.
        """
        logger.info(
            f"streaming query — thread={thread_id}, user={user_id}, "
            f"q='{question[:80]}'"
        )
        config = self._build_config(thread_id, user_id)

        # Nodes whose LLM output is the final answer we want to stream
        _ANSWER_NODES = {"generate_answer", "generate_direct"}

        try:
            async for msg, metadata in self._graph.astream(
                self._initial_state(question),
                config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node in _ANSWER_NODES and hasattr(msg, "content") and msg.content:
                    yield msg.content
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"\n\n[Error: {type(e).__name__}: {str(e)}]"

    def health_check(self) -> bool:
        return self._graph is not None

    @staticmethod
    def _format_result(state: dict) -> dict:
        good_docs = state.get("good_docs", []) or []
        web_docs = state.get("web_docs", []) or []

        sources = [
            {
                "content": (
                    d.page_content[:500] + "..."
                    if len(d.page_content) > 500
                    else d.page_content
                ),
                "metadata": d.metadata,
                "origin": "internal",
            }
            for d in good_docs
        ] + [
            {
                "content": (
                    d.page_content[:500] + "..."
                    if len(d.page_content) > 500
                    else d.page_content
                ),
                "metadata": d.metadata,
                "origin": "web",
            }
            for d in web_docs
        ]

        return {
            "answer": state.get("answer", ""),
            "sources": sources,
            "crag_verdict": state.get("crag_verdict", ""),
            "crag_reason": state.get("crag_reason", ""),
            "issup": state.get("issup", ""),
            "evidence": state.get("evidence", []),
            "isuse": state.get("isuse", ""),
            "use_reason": state.get("use_reason", ""),
            "retries": state.get("retries", 0),
            "rewrite_tries": state.get("rewrite_tries", 0),
        }

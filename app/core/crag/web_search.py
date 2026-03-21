"""CRAG web-search module — Tavily integration.

Handles both:
  1. Query rewriting — converts the user question into a focused web-search query.
  2. Web document fetching — calls Tavily and wraps results as LangChain Documents.
"""

from functools import lru_cache

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Query rewrite schema + prompt
# ------------------------------------------------------------------

class WebQuery(BaseModel):
    """Rewritten web-search query."""

    query: str = Field(
        ...,
        description="Focused web-search query (6–14 keywords).",
    )


_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a focused web-search query.\n"
            "Rules:\n"
            "  - Keep it short: 6–14 keywords.\n"
            "  - If the question implies recency (recent/latest/last week), "
            "add a time constraint such as '(last 30 days)'.\n"
            "  - Do NOT answer the question.\n"
            "  - Return JSON with a single key: query.",
        ),
        ("human", "Question: {question}"),
    ]
)


# ------------------------------------------------------------------
# Service class
# ------------------------------------------------------------------

class WebSearchService:
    """Rewrites queries and fetches web documents via Tavily."""

    def __init__(self) -> None:
        settings = get_settings()

        llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
        )
        self._rewrite_chain = _REWRITE_PROMPT | llm.with_structured_output(WebQuery)

        self._tavily = TavilySearchResults(
            max_results=settings.tavily_max_results,
            tavily_api_key=settings.tavily_api_key,
        )
        logger.info(
            f"WebSearchService ready — "
            f"tavily max_results={settings.tavily_max_results}"
        )

    def rewrite_query(self, question: str) -> str:
        """Rewrite a question into an optimised web-search query.

        Args:
            question: Original user question.

        Returns:
            Rewritten query string.
        """
        logger.debug(f"Rewriting query for: {question[:80]}")
        result: WebQuery = self._rewrite_chain.invoke({"question": question})
        logger.info(f"Rewritten web query: {result.query}")
        return result.query

    def search(self, query: str) -> list[Document]:
        """Execute a Tavily web search and return results as Documents.

        Args:
            query: Search query string.

        Returns:
            List of :class:`Document` objects with content and metadata.
        """
        logger.info(f"Tavily web search: '{query}'")
        try:
            results = self._tavily.invoke({"query": query})
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

        web_docs: list[Document] = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(
                Document(
                    page_content=text,
                    metadata={"title": title, "url": url, "source": url},
                )
            )

        logger.info(f"Tavily returned {len(web_docs)} results")
        return web_docs


@lru_cache
def get_web_search_service() -> WebSearchService:
    """Return a cached :class:`WebSearchService` instance."""
    return WebSearchService()

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SourceDocument,
)
from app.core.csrag_engine import CSRAGEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


def get_engine(request: Request) -> CSRAGEngine:
    return request.app.state.engine


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Ask a question",
    description=(
        "Submit a question and receive an AI-generated answer produced by the "
        "full CSRAG pipeline: CRAG retrieval quality verification, SRAG answer "
        "grounding verification, short-term memory summarisation, and long-term "
        "memory personalisation."
    ),
)
async def chat(
    body: ChatRequest,
    engine: CSRAGEngine = Depends(get_engine),
) -> ChatResponse:
    logger.info(
        f"Chat — thread={body.thread_id}, user={body.user_id}, "
        f"q='{body.question[:80]}'"
    )
    start_time = time.time()

    try:
        result = await engine.aquery(
            question=body.question,
            thread_id=body.thread_id,
            user_id=body.user_id,
        )
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}",
        )

    processing_time = (time.time() - start_time) * 1000

    sources: list[SourceDocument] | None = None
    if body.include_sources:
        sources = [
            SourceDocument(
                content=s["content"],
                metadata=s["metadata"],
                origin=s["origin"],
            )
            for s in result.get("sources", [])
        ]

    return ChatResponse(
        question=body.question,
        answer=result["answer"],
        sources=sources,
        processing_time_ms=round(processing_time, 2),
        crag_verdict=result.get("crag_verdict", ""),
        crag_reason=result.get("crag_reason", ""),
        issup=result.get("issup", ""),
        evidence=result.get("evidence", []),
        isuse=result.get("isuse", ""),
        use_reason=result.get("use_reason", ""),
        retries=result.get("retries", 0),
        rewrite_tries=result.get("rewrite_tries", 0),
    )


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Streaming error"},
    },
    summary="Ask a question (streaming)",
    description="Stream the CSRAG answer token by token as plain text.",
)
async def chat_stream(
    body: ChatRequest,
    engine: CSRAGEngine = Depends(get_engine),
) -> StreamingResponse:
    logger.info(
        f"Chat stream — thread={body.thread_id}, user={body.user_id}, "
        f"q='{body.question[:80]}'"
    )

    async def generate():
        try:
            async for chunk in engine.astream(
                question=body.question,
                thread_id=body.thread_id,
                user_id=body.user_id,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")

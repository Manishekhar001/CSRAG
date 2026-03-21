from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from app import __version__
from app.api.schemas import HealthResponse, ReadinessResponse
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


def get_vector_store(request: Request) -> VectorStoreService:
    return request.app.state.vector_store


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns service status and version. No external dependencies checked.",
)
async def health_check() -> HealthResponse:
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=__version__,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Checks Qdrant and Postgres connectivity.",
)
async def readiness_check(
    request: Request,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> ReadinessResponse:
    logger.debug("Readiness check requested")

    qdrant_ok = vector_store.health_check()
    if not qdrant_ok:
        raise HTTPException(status_code=503, detail="Qdrant is not reachable")

    postgres_ok = False
    try:
        store = request.app.state.store
        store.search(("__health__",))
        postgres_ok = True
    except Exception as e:
        logger.error(f"Postgres health check failed: {e}")

    if not postgres_ok:
        raise HTTPException(status_code=503, detail="Postgres is not reachable")

    collection_info = vector_store.get_collection_info()

    return ReadinessResponse(
        status="ready",
        qdrant_connected=qdrant_ok,
        postgres_connected=postgres_ok,
        collection_info=collection_info,
    )

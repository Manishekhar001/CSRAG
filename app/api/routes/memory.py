"""Long-term memory management endpoints."""

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import DeleteMemoryResponse, ErrorResponse, MemoryItem, MemoryListResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


def get_store(request: Request):
    """Dependency: pull the shared PostgresStore from app.state."""
    return request.app.state.store


@router.get(
    "/{user_id}",
    response_model=MemoryListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Postgres error"},
    },
    summary="List user memories",
    description="Return all long-term memory facts stored for a given user.",
)
async def list_memories(user_id: str, request: Request) -> MemoryListResponse:
    """Retrieve all LTM facts for a user from Postgres."""
    logger.debug(f"Listing memories for user={user_id}")
    store = get_store(request)

    try:
        ns = ("user", user_id, "details")
        items = store.search(ns)
        memories = [MemoryItem(data=it.value.get("data", "")) for it in items]
        return MemoryListResponse(
            user_id=user_id,
            memories=memories,
            count=len(memories),
        )
    except Exception as e:
        logger.error(f"list_memories error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memories: {str(e)}",
        )


@router.delete(
    "/{user_id}",
    response_model=DeleteMemoryResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Postgres error"},
    },
    summary="Clear user memories",
    description="Delete all long-term memory facts for a given user. This is irreversible.",
)
async def delete_memories(user_id: str, request: Request) -> DeleteMemoryResponse:
    """Delete all LTM facts for a user from Postgres."""
    logger.warning(f"Deleting all memories for user={user_id}")
    store = get_store(request)

    try:
        ns = ("user", user_id, "details")
        items = store.search(ns)
        for item in items:
            store.delete(ns, item.key)

        logger.info(f"Deleted {len(items)} memories for user={user_id}")
        return DeleteMemoryResponse(
            message=f"Deleted {len(items)} memories successfully",
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"delete_memories error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete memories: {str(e)}",
        )

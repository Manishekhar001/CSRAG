"""Pydantic schemas for all API request/response models."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ======================================================================
# Health
# ======================================================================

class HealthResponse(BaseModel):
    """Basic health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp",
    )
    version: str = Field(..., description="Application version")


class ReadinessResponse(BaseModel):
    """Readiness check including dependency connectivity."""

    status: str = Field(..., description="Service status")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")
    postgres_connected: bool = Field(..., description="Postgres connection status")
    collection_info: dict = Field(..., description="Qdrant collection metadata")


# ======================================================================
# Documents
# ======================================================================

class DocumentUploadResponse(BaseModel):
    """Response after a successful document upload."""

    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of chunks indexed")
    document_ids: list[str] = Field(..., description="Generated chunk IDs")


class CollectionInfoResponse(BaseModel):
    """Qdrant collection metadata."""

    collection_name: str = Field(..., description="Collection name")
    total_documents: int = Field(..., description="Total indexed chunks")
    status: str = Field(..., description="Collection status")


# ======================================================================
# Chat / Query
# ======================================================================

class ChatRequest(BaseModel):
    """Request body for a CSRAG chat query."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
    )
    thread_id: str = Field(
        ...,
        description=(
            "Conversation thread ID. Use the same ID across turns to maintain "
            "short-term memory. Create a new UUID for a fresh conversation."
        ),
    )
    user_id: str = Field(
        ...,
        description=(
            "User identifier. Long-term memory is keyed on this. "
            "Must be consistent across sessions for the same user."
        ),
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in the response.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the refund policy?",
                    "thread_id": "thread-abc-123",
                    "user_id": "user-xyz-456",
                    "include_sources": True,
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """A single source document returned with the answer."""

    content: str = Field(..., description="Truncated document content (max 500 chars)")
    metadata: dict[str, Any] = Field(..., description="Document metadata")
    origin: Literal["internal", "web"] = Field(
        ...,
        description="Whether this came from Qdrant or a web search.",
    )


class ChatResponse(BaseModel):
    """Full CSRAG response including verification metadata."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] | None = Field(
        None,
        description="Source documents (internal + web) used to generate the answer",
    )
    processing_time_ms: float = Field(..., description="End-to-end latency in ms")

    # CRAG metadata
    crag_verdict: str = Field(
        "",
        description="CRAG retrieval quality verdict: CORRECT | AMBIGUOUS | INCORRECT",
    )
    crag_reason: str = Field("", description="CRAG verdict justification")

    # SRAG metadata
    issup: str = Field(
        "",
        description="SRAG support verdict: fully_supported | partially_supported | no_support",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence quotes from the context supporting the answer",
    )
    isuse: str = Field(
        "",
        description="SRAG usefulness verdict: useful | not_useful",
    )
    use_reason: str = Field("", description="SRAG usefulness justification")

    # Loop counters
    retries: int = Field(0, description="Number of answer revision iterations")
    rewrite_tries: int = Field(0, description="Number of question rewrite iterations")


# ======================================================================
# Memory
# ======================================================================

class MemoryItem(BaseModel):
    """A single long-term memory fact."""

    data: str = Field(..., description="Stored memory text")


class MemoryListResponse(BaseModel):
    """All LTM memories for a given user."""

    user_id: str = Field(..., description="User identifier")
    memories: list[MemoryItem] = Field(..., description="List of stored facts")
    count: int = Field(..., description="Total number of facts")


class DeleteMemoryResponse(BaseModel):
    """Response after clearing a user's LTM."""

    message: str = Field(..., description="Status message")
    user_id: str = Field(..., description="User whose memories were cleared")


# ======================================================================
# Errors
# ======================================================================

class ErrorResponse(BaseModel):
    """Generic error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional detail")

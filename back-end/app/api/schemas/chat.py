"""Pydantic schemas for chat endpoints."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    """Request body for creating a new chat session."""

    user_id: str | None = Field(
        default=None,
        description="Optional caller-supplied user identifier",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs attached to the session",
    )


class SessionResponse(BaseModel):
    """Returned after a session is created."""

    session_id: UUID = Field(default_factory=uuid4)
    user_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    """Request body for sending a message to the chatbot."""

    session_id: UUID = Field(..., description="ID of an existing chat session")
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User message text (Vietnamese or English)",
    )
    stream: bool = Field(
        default=False,
        description="If true, the response is streamed via SSE",
    )


class SourceDocument(BaseModel):
    """A product snippet retrieved from the vector store and used as RAG context."""

    product_id: str
    product_name: str
    score: float = Field(ge=0.0, le=1.0)
    snippet: str | None = None


class ChatResponse(BaseModel):
    """Response returned after processing a user message."""

    session_id: UUID
    message_id: UUID = Field(default_factory=uuid4)
    role: Role = Role.ASSISTANT
    content: str
    source_documents: list[SourceDocument] = Field(default_factory=list)
    model: str
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage stats from the LLM (input_tokens, output_tokens)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """A single turn in the conversation history."""

    message_id: UUID
    role: Role
    content: str
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Paginated conversation history for a session."""

    session_id: UUID
    messages: list[HistoryMessage]
    total: int
    page: int
    page_size: int

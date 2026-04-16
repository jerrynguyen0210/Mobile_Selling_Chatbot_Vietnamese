"""Chat endpoints: session management, message sending, history retrieval."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.api.schemas.chat import (
    ChatHistoryResponse,
    ChatResponse,
    HistoryMessage,
    MessageRequest,
    Role,
    SessionCreate,
    SessionResponse,
)
from app.dependencies import AppSettings, RedisClient
from app.services import ChatOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@router.post(
    "/session",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_session(
    body: SessionCreate,
    redis: RedisClient,
    settings: AppSettings,
) -> SessionResponse:
    """
    Creates a new conversation session.  The returned ``session_id`` must be
    included in every subsequent ``POST /message`` request.
    """
    session = SessionResponse(user_id=body.user_id, metadata=body.metadata)

    # Persist session metadata in Redis for fast look-up
    if settings.enable_cache:
        key = f"session:{session.session_id}"
        await redis.hset(  # type: ignore[misc]
            key,
            mapping={
                "user_id": session.user_id or "",
                "created_at": session.created_at.isoformat(),
            },
        )
        await redis.expire(key, settings.session_ttl)

    # TODO: persist to long-term storage once ORM models are in place
    logger.info("Created session %s (user=%s)", session.session_id, session.user_id)
    return session


# ---------------------------------------------------------------------------
# Send message
# ---------------------------------------------------------------------------


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message and receive an AI response",
)
async def send_message(
    body: MessageRequest,
    redis: RedisClient,
    settings: AppSettings,
) -> ChatResponse:
    """
    Accepts a user message, runs the RAG pipeline, calls the Claude LLM, and
    returns the assistant reply together with any retrieved source documents.

    Set ``stream=true`` to receive a streaming SSE response instead (handled by
    the WebSocket / SSE layer — this endpoint returns 200 with the full payload
    when streaming is disabled).
    """
    # Validate session exists
    if settings.enable_cache:
        session_key = f"session:{body.session_id}"
        exists = await redis.exists(session_key)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {body.session_id} not found or expired.",
            )

    logger.info("Message received for session %s", body.session_id)
    orchestrator = ChatOrchestrator(settings=settings, redis=redis)
    return await orchestrator.process(body)


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Retrieve conversation history for a session",
)
async def get_history(
    session_id: UUID,
    settings: AppSettings,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> ChatHistoryResponse:
    """
    Returns a paginated list of all messages exchanged in the given session,
    ordered oldest-first.
    """
    if not settings.enable_conversation_history:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Conversation history is disabled.",
        )

    # TODO: replace stub with real DB query once ORM models exist
    logger.info("History requested for session %s (page=%d)", session_id, page)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(
                message_id=session_id,  # placeholder — same UUID reused
                role=Role.ASSISTANT,
                content="(history not yet persisted)",
                created_at=__import__("datetime").datetime.utcnow(),
            )
        ],
        total=1,
        page=page,
        page_size=page_size,
    )

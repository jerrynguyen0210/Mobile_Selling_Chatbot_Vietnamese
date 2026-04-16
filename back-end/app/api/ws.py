"""WebSocket endpoint for real-time chat with typing indicators."""

import asyncio
import json
import logging
from enum import StrEnum
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


# ---------------------------------------------------------------------------
# Message envelope types
# ---------------------------------------------------------------------------


class ClientEventType(StrEnum):
    MESSAGE = "message"
    PING = "ping"


class ServerEventType(StrEnum):
    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"
    CHUNK = "chunk"          # streaming token
    RESPONSE = "response"    # full response (non-streaming)
    ERROR = "error"
    PONG = "pong"


class ClientMessage(BaseModel):
    type: ClientEventType
    session_id: UUID
    text: str | None = None


class ServerMessage(BaseModel):
    type: ServerEventType
    session_id: UUID | None = None
    content: str | None = None
    metadata: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Tracks active WebSocket connections keyed by session_id."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[session_id] = websocket
        logger.info("WS connected: session=%s  total=%d", session_id, len(self._connections))

    def disconnect(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
        logger.info("WS disconnected: session=%s  total=%d", session_id, len(self._connections))

    async def send(self, session_id: str, message: ServerMessage) -> None:
        ws = self._connections.get(session_id)
        if ws:
            await ws.send_text(message.model_dump_json())

    async def broadcast(self, message: ServerMessage) -> None:
        dead: list[str] = []
        for sid, ws in list(self._connections.items()):
            try:
                await ws.send_text(message.model_dump_json())
            except Exception:
                dead.append(sid)
        for sid in dead:
            self.disconnect(sid)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _stream_llm_response(session_id: UUID, user_text: str) -> str:
    """
    Placeholder streaming LLM call.

    Replace this with a real ``anthropic.AsyncAnthropic().messages.stream()``
    call once the service layer is wired up.  Each yielded chunk is sent to the
    client as a ``ServerEventType.CHUNK`` frame.
    """
    sid = str(session_id)

    # Typing indicator → on
    await manager.send(
        sid,
        ServerMessage(type=ServerEventType.TYPING_START, session_id=session_id),
    )

    # TODO: replace with real streaming call
    placeholder_tokens = [
        "Xin ", "chào! ", "Tôi ", "có thể ", "giúp ", "bạn ",
        "tìm ", "điện thoại ", "phù hợp. ",
    ]
    full_response = ""
    for token in placeholder_tokens:
        await asyncio.sleep(0.05)  # simulate latency between tokens
        full_response += token
        await manager.send(
            sid,
            ServerMessage(
                type=ServerEventType.CHUNK,
                session_id=session_id,
                content=token,
            ),
        )

    # Typing indicator → off
    await manager.send(
        sid,
        ServerMessage(type=ServerEventType.TYPING_STOP, session_id=session_id),
    )

    return full_response


# ---------------------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------------------


@router.websocket("/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: UUID) -> None:
    """
    Real-time chat endpoint.

    **Protocol (JSON frames):**

    Client → Server:
    ```json
    {"type": "message", "session_id": "<uuid>", "text": "Tìm điện thoại dưới 10 triệu"}
    {"type": "ping",    "session_id": "<uuid>"}
    ```

    Server → Client:
    ```json
    {"type": "typing_start", "session_id": "<uuid>"}
    {"type": "chunk",        "session_id": "<uuid>", "content": "token..."}
    {"type": "typing_stop",  "session_id": "<uuid>"}
    {"type": "response",     "session_id": "<uuid>", "content": "full reply", "metadata": {...}}
    {"type": "pong",         "session_id": "<uuid>"}
    {"type": "error",        "session_id": "<uuid>", "content": "error message"}
    ```
    """
    sid = str(session_id)
    await manager.connect(sid, websocket)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                event = ClientMessage(**json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                await manager.send(
                    sid,
                    ServerMessage(
                        type=ServerEventType.ERROR,
                        session_id=session_id,
                        content=f"Invalid message format: {exc}",
                    ),
                )
                continue

            if event.type == ClientEventType.PING:
                await manager.send(
                    sid, ServerMessage(type=ServerEventType.PONG, session_id=session_id)
                )
                continue

            if event.type == ClientEventType.MESSAGE:
                if not event.text:
                    await manager.send(
                        sid,
                        ServerMessage(
                            type=ServerEventType.ERROR,
                            session_id=session_id,
                            content="Message text is required.",
                        ),
                    )
                    continue

                full_response = await _stream_llm_response(session_id, event.text)

                # Send the complete response as a final frame (useful for
                # clients that want one atomic payload rather than chunks)
                await manager.send(
                    sid,
                    ServerMessage(
                        type=ServerEventType.RESPONSE,
                        session_id=session_id,
                        content=full_response,
                        metadata={"model": get_settings().claude_model},
                    ),
                )

    except WebSocketDisconnect:
        manager.disconnect(sid)
    except Exception as exc:
        logger.exception("Unexpected WebSocket error for session %s: %s", sid, exc)
        try:
            await manager.send(
                sid,
                ServerMessage(
                    type=ServerEventType.ERROR,
                    session_id=session_id,
                    content="Internal server error.",
                ),
            )
        except Exception:
            pass
        manager.disconnect(sid)

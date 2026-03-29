"""Central chat orchestrator: NLU → intent classification → handler routing → response."""

import logging

import redis.asyncio as aioredis
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.chat import ChatResponse, MessageRequest, SourceDocument
from app.config import Settings
from app.exceptions import LLMError
from app.nlu import HANDLER_FOR, IntentClassifier, NLUResult, handle_unknown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SALES_ASSISTANT_SYSTEM_PROMPT = """\
Bạn là trợ lý tư vấn bán điện thoại di động thông minh, thân thiện và chuyên nghiệp \
phục vụ thị trường Việt Nam.

Nhiệm vụ:
- Tư vấn sản phẩm phù hợp với nhu cầu và ngân sách khách hàng
- Cung cấp thông tin chi tiết, chính xác về sản phẩm
- Hỗ trợ đặt hàng, theo dõi và huỷ đơn
- Trả lời câu hỏi về chính sách bảo hành, đổi trả, thanh toán, giao hàng

Quy tắc:
- Luôn trả lời bằng tiếng Việt trừ khi khách hàng dùng tiếng Anh
- Giọng điệu thân thiện, lịch sự, chuyên nghiệp
- Không bịa đặt thông tin sản phẩm hay giá cả; nếu thiếu dữ liệu hãy cho biết
- Khi context sản phẩm được cung cấp ở phần đầu tin nhắn, hãy ưu tiên dùng đó\
"""


# ---------------------------------------------------------------------------
# Prompt templates (module-level, reused across requests)
# ---------------------------------------------------------------------------

# MessagesPlaceholder injects per-session turn history between the system
# prompt and the current human message.
_REPLY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder("history"),
        ("human", "{content}"),
    ]
)


# ---------------------------------------------------------------------------
# Process-scoped short-term memory store
#
# Lives at module level so it survives per-request ChatOrchestrator instances.
# Keys are session_id strings; values are InMemoryChatMessageHistory objects
# that accumulate HumanMessage / AIMessage pairs for the session lifetime.
# ---------------------------------------------------------------------------

_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def clear_session_history(session_id: str) -> None:
    """Remove stored history for *session_id* (call on session expiry / logout)."""
    _session_histories.pop(session_id, None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ChatOrchestrator:
    """
    Central brain of the chat pipeline.

    Flow:
        1. ``_classify_intent`` — NLU chain (prompt | llm.with_structured_output)
        2. ``_route``           — dispatch to the matching intent handler
        3. handler              — optionally inject RAG context, then invoke the
                                  history-aware reply chain
        4. return ``ChatResponse``

    Model flexibility
    -----------------
    Pass any ``BaseChatModel`` (OpenAI, Gemini, Ollama, …) via *nlu_llm* /
    *reply_llm*.  When omitted the orchestrator builds ``ChatAnthropic``
    instances from *settings*, preserving backward compatibility.

    Short-term memory
    -----------------
    ``RunnableWithMessageHistory`` wraps the reply chain and automatically
    appends each human turn + AI response to the per-session
    ``InMemoryChatMessageHistory`` stored in ``_session_histories``.
    """

    def __init__(
        self,
        settings: Settings,
        redis: aioredis.Redis | None = None,  # type: ignore[type-arg]
        db: AsyncSession | None = None,
        *,
        nlu_llm: BaseChatModel | None = None,
        reply_llm: BaseChatModel | None = None,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._db = db

        # ------------------------------------------------------------------
        # Resolve LLMs — caller may inject any BaseChatModel; default to
        # ChatAnthropic derived from settings.
        # ------------------------------------------------------------------
        _nlu_llm: BaseChatModel = nlu_llm or ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
            max_tokens=256,
        )
        _reply_llm: BaseChatModel = reply_llm or ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
            max_tokens=settings.claude_max_tokens,
        )

        # NLU: single-turn, no history needed.
        self._classifier = IntentClassifier(_nlu_llm)

        # Reply: history-aware chain — RunnableWithMessageHistory loads /
        # saves turns in _session_histories keyed by session_id.
        self._reply_chain = RunnableWithMessageHistory(
            _REPLY_PROMPT | _reply_llm,
            _get_session_history,
            input_messages_key="content",
            history_messages_key="history",
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process(self, request: MessageRequest) -> ChatResponse:
        """Run the full pipeline for one user turn and return the assistant reply."""
        nlu = await self._classify_intent(request.message)
        logger.info(
            "session=%s intent=%s confidence=%.2f entities=%s",
            request.session_id,
            nlu.intent,
            nlu.confidence,
            nlu.entities,
        )
        return await self._route(request, nlu)

    # -------------------------------------------------------------------------
    # NLU — intent classification
    # -------------------------------------------------------------------------

    async def _classify_intent(self, message: str) -> NLUResult:
        """Delegate to IntentClassifier; fallback to UNKNOWN is handled there."""
        return await self._classifier.classify(message)

    # -------------------------------------------------------------------------
    # Router
    # -------------------------------------------------------------------------

    async def _route(self, request: MessageRequest, nlu: NLUResult) -> ChatResponse:
        """Dispatch *request* to the handler that matches *nlu.intent*."""
        handler = HANDLER_FOR.get(nlu.intent, handle_unknown)
        return await handler(request, nlu, self._llm_reply, self._build_response)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    async def _llm_reply(
        self,
        session_id: str,
        user_message: str,
        *,
        system: str = _SALES_ASSISTANT_SYSTEM_PROMPT,
        extra_context: str = "",
    ) -> tuple[str, dict[str, int]]:
        """
        Invoke the history-aware reply chain for one turn.

        *extra_context* (RAG snippets, order metadata, …) is prepended to the
        user message so the model has grounded facts to draw on.

        Returns ``(reply_text, usage_dict)``.
        """
        content = (
            f"[Thông tin ngữ cảnh]\n{extra_context}\n\n---\n\nKhách hàng: {user_message}"
            if extra_context
            else user_message
        )
        try:
            ai_msg: AIMessage = await self._reply_chain.ainvoke(
                {"system_prompt": system, "content": content},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as exc:
            raise LLMError(f"LLM reply failed: {exc}") from exc

        usage_raw: dict[str, int] = ai_msg.response_metadata.get("usage", {})
        usage = {
            "input_tokens": usage_raw.get("input_tokens", 0),
            "output_tokens": usage_raw.get("output_tokens", 0),
        }
        return str(ai_msg.content), usage

    def _build_response(
        self,
        request: MessageRequest,
        content: str,
        usage: dict[str, int],
        source_documents: list[SourceDocument] | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            session_id=request.session_id,
            content=content,
            source_documents=source_documents or [],
            model=self._settings.claude_model,
            usage=usage,
        )


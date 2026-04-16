"""Intent classification and intent-level handlers.

All intent logic lives here so that adding a new intent requires changes in
exactly one file: add to ``Intent``, write ``handle_<intent>``, register in
``HANDLER_FOR``.
"""

import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.api.schemas.chat import ChatResponse, MessageRequest
from app.config import Settings, get_settings
from app.rag.retriever import get_retriever

logger = logging.getLogger(__name__)

# Type aliases for the two orchestrator helpers injected into every handler.
LLMReply = Callable[..., Awaitable[tuple[str, dict[str, int]]]]
BuildResponse = Callable[..., ChatResponse]


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------


class Intent(StrEnum):
    PRODUCT_SEARCH = "product_search"       # tìm kiếm theo tiêu chí
    PRODUCT_DETAIL = "product_detail"       # thông tin chi tiết sản phẩm
    PRODUCT_COMPARE = "product_compare"     # so sánh nhiều sản phẩm
    GREETING = "greeting"                   # chào hỏi
    CHITCHAT = "chitchat"                   # ngoài lề
    UNKNOWN = "unknown"                     # không xác định


class NLUResult(BaseModel):
    """Structured output from the intent-classification step."""

    intent: Intent
    confidence: float
    entities: dict[str, Any]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_NLU_SYSTEM_PROMPT = """\
Bạn là bộ phân loại ý định (intent classifier) cho chatbot bán điện thoại di động \
tại thị trường Việt Nam.

Phân tích tin nhắn người dùng và trả về JSON với định dạng:
{{
  "intent": "<intent>",
  "confidence": <float 0.0–1.0>,
  "entities": {{<key>: <value>}}
}}

Danh sách intent:
- product_search   : tìm điện thoại theo tiêu chí (hãng, giá, RAM, màn hình…)
- product_detail   : hỏi thông tin chi tiết về một sản phẩm cụ thể
- product_compare  : so sánh hai hoặc nhiều sản phẩm
- greeting         : chào hỏi, bắt đầu cuộc trò chuyện
- chitchat         : chủ đề ngoài lề không liên quan đến điện thoại
- unknown          : không xác định được ý định

Entities thường gặp (chỉ trích xuất khi có):
  brand, model, price_min, price_max, ram_gb, storage_gb, os,
  product_ids (list of strings)

Chỉ trả về JSON, không thêm giải thích hay markdown.\
"""

_NLU_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _NLU_SYSTEM_PROMPT),
        ("human", "{message}"),
    ]
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Stateless NLU classifier wrapping an LLM chain.

    The caller provides any ``BaseChatModel``; the classifier builds a
    single-turn chain (no history) and exposes one async method: ``classify``.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = _NLU_PROMPT | llm.with_structured_output(NLUResult)

    async def classify(self, message: str) -> NLUResult:
        """Classify *message*; falls back to ``Intent.UNKNOWN`` on any error."""
        try:
            result: NLUResult = await self._chain.ainvoke({"message": message})
            return result
        except Exception as exc:
            logger.warning(
                "NLU classification failed — falling back to UNKNOWN. error=%s", exc
            )
            return NLUResult(intent=Intent.UNKNOWN, confidence=0.0, entities={})


# ---------------------------------------------------------------------------
# Intent handlers
#
# Each handler is a plain async function.  The orchestrator injects its own
# ``llm_reply`` and ``build_response`` helpers so handlers stay stateless and
# dependency-free.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_product_search_settings() -> Settings:
    return get_settings().model_copy(update={"retrieval_score_threshold": 0.3})


async def handle_product_search(
    request: MessageRequest,
    nlu: NLUResult,
    llm_reply: LLMReply,
    build_response: BuildResponse,
) -> ChatResponse:
    # TODO: embed request.message → search Qdrant → inject top-K snippets as extra_context
    # entities_summary = ", ".join(f"{k}={v}" for k, v in nlu.entities.items())
    # extra = f"Tiêu chí tìm kiếm (trích từ NLU): {entities_summary}" if entities_summary else ""
    settings = _get_product_search_settings()
    retriever = get_retriever(settings)
    docs, context = await retriever.search_for_context(request.message, top_k=1)
    content, usage = await llm_reply(str(request.session_id), request.message, extra_context=context)
    return build_response(request, content, usage)


async def handle_product_detail(
    request: MessageRequest,
    nlu: NLUResult,
    llm_reply: LLMReply,
    build_response: BuildResponse,
) -> ChatResponse:
    # TODO: fetch product by nlu.entities.get("model") or "product_ids"[0] from DB/Qdrant
    settings = _get_product_search_settings()
    retriever = get_retriever(settings)
    docs = await retriever.search(request.message)
    logger.info("Product detail search returned %d docs for query=%s", len(docs), docs[0])
    content, usage = await llm_reply(str(request.session_id), request.message, extra_context=docs[0])
    return build_response(request, content, usage)


async def handle_product_compare(
    request: MessageRequest,
    nlu: NLUResult,
    llm_reply: LLMReply,
    build_response: BuildResponse,
) -> ChatResponse:
    # TODO: fetch each product in nlu.entities.get("product_ids", []) and build
    #       a structured comparison table to inject as extra_context
    content, usage = await llm_reply(str(request.session_id), request.message)
    return build_response(request, content, usage)


async def handle_greeting(
    request: MessageRequest,
    nlu: NLUResult,
    llm_reply: LLMReply,
    build_response: BuildResponse,
) -> ChatResponse:
    content, usage = await llm_reply(str(request.session_id), request.message)
    return build_response(request, content, usage)


async def handle_chitchat(
    request: MessageRequest,
    nlu: NLUResult,
    llm_reply: LLMReply,
    build_response: BuildResponse,
) -> ChatResponse:
    content, usage = await llm_reply(str(request.session_id), request.message)
    return build_response(request, content, usage)


async def handle_unknown(
    request: MessageRequest,
    nlu: NLUResult,
    llm_reply: LLMReply,
    build_response: BuildResponse,
) -> ChatResponse:
    content = (
        "Xin lỗi, tôi chưa hiểu rõ yêu cầu của bạn. "
        "Bạn có thể mô tả lại cụ thể hơn không? "
        "Tôi có thể giúp bạn tìm điện thoại, xem thông tin sản phẩm, "
        "và so sánh các mẫu máy."
    )
    return build_response(request, content, usage={"input_tokens": 0, "output_tokens": 0})


# ---------------------------------------------------------------------------
# Dispatch table  (Intent → handler function)
#
# Adding a new intent: extend Intent above, write handle_<intent>, add a row.
# ---------------------------------------------------------------------------

HandlerFn = Callable[..., Awaitable[ChatResponse]]

HANDLER_FOR: dict[Intent, HandlerFn] = {
    Intent.PRODUCT_SEARCH:  handle_product_search,
    Intent.PRODUCT_DETAIL:  handle_product_detail,
    Intent.PRODUCT_COMPARE: handle_product_compare,
    Intent.GREETING:        handle_greeting,
    Intent.CHITCHAT:        handle_chitchat,
    Intent.UNKNOWN:         handle_unknown,
}

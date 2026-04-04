"""Semantic retriever: embeds a query, searches Qdrant, and applies
metadata filters (brand, price range, stock).

Architecture
------------
* ``ProductRetriever`` – main class, suitable for use as a FastAPI singleton
  via the ``get_retriever`` factory below.
* ``SearchFilters``    – dataclass carrying all optional filter parameters.

Filter implementation
---------------------
brand       Qdrant-level ``MatchText`` condition on the ``title`` payload
            field.  Tokenised full-text match, so "Samsung" matches any title
            containing the word regardless of capitalisation.

min/max_price
            Post-retrieval filter.  Prices are stored as Vietnamese formatted
            strings (e.g. "5.990.000 ₫"); ``_parse_price_vnd`` strips
            non-digits for numeric comparison.

in_stock    Post-retrieval filter on the ``in_stock`` boolean payload field.
            Documents that lack the field are treated as *unknown* and are
            excluded when ``in_stock=True`` is requested.

            .. note::
                The current ingestion script does not populate ``in_stock``.
                Add it to the CSV schema and re-ingest to activate this filter.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchText

from app.api.schemas.chat import SourceDocument
from app.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


@dataclass
class SearchFilters:
    """Optional metadata filters for :meth:`ProductRetriever.search`."""

    brand: str | None = None
    """Filter by brand name (case-insensitive word match inside *title*).

    Examples: ``"Samsung"``, ``"iPhone"``, ``"Xiaomi"``, ``"OPPO"``.
    """

    min_price: float | None = None
    """Minimum price in VND (inclusive).  ``None`` → no lower bound."""

    max_price: float | None = None
    """Maximum price in VND (inclusive).  ``None`` → no upper bound."""

    in_stock: bool | None = None
    """``True`` → in-stock only.  ``False`` → out-of-stock only.
    ``None`` → no stock filter (default).

    Requires an ``in_stock`` boolean field in the Qdrant payload.
    Documents that lack the field are excluded when this is ``True``.
    """


# ---------------------------------------------------------------------------
# Internal hit representation
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    point_id: str
    score: float
    title: str
    current_price: str
    price_vnd: float | None
    url: str
    product_promotion: str
    product_specs: str
    color_options: str
    in_stock: bool | None  # None when the payload field is absent


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class ProductRetriever:
    """Semantic search over the ``mobile_products`` Qdrant collection.

    Typical usage inside an async handler::

        retriever = get_retriever(settings)
        docs, context = await retriever.search_for_context(
            "điện thoại Samsung giá rẻ",
            filters=SearchFilters(brand="Samsung", max_price=8_000_000),
        )
        # Pass *context* to the LLM as extra_context in _llm_reply().

    Parameters
    ----------
    settings:
        Application settings (Qdrant URL/key, collection name, embedding
        model path, top-K, and score threshold are all read from here).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._top_k = settings.retrieval_top_k
        self._threshold = settings.retrieval_score_threshold

        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

        # SentenceTransformer download / load happens once at startup.
        logger.info("Loading embedding model: %s", settings.embedding_model)
        # Import deferred to avoid a slow import at module load time.
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(settings.embedding_model)
        logger.info("ProductRetriever ready  collection=%s", self._collection)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SourceDocument]:
        """Embed *query*, search Qdrant, apply post-filters, return results.

        Parameters
        ----------
        query:
            Natural-language search query (Vietnamese or English).
        filters:
            Optional :class:`SearchFilters`.  ``None`` → no filters applied.
        top_k:
            Override ``settings.retrieval_top_k``.
        score_threshold:
            Override ``settings.retrieval_score_threshold``.

        Returns
        -------
        list[SourceDocument]
            Up to *top_k* results sorted by descending cosine similarity,
            with all requested filters applied.
        """
        f = filters or SearchFilters()
        k = top_k if top_k is not None else self._top_k
        threshold = score_threshold if score_threshold is not None else self._threshold

        # Over-fetch to absorb post-filter attrition from price / stock filters.
        needs_post_filter = (
            f.min_price is not None
            or f.max_price is not None
            or f.in_stock is not None
        )
        fetch_limit = k * 5 if needs_post_filter else k

        # SentenceTransformer.encode is CPU-bound / blocking — run off the loop.
        query_vector: list[float] = await asyncio.to_thread(self._embed, query)

        # Run the sync QdrantClient call off the event loop to keep it
        # non-blocking.  functools.partial lets us pass keyword args to
        # asyncio.to_thread without a lambda.
        response = await asyncio.to_thread(
            functools.partial(
                self._client.query_points,
                collection_name=self._collection,
                query=query_vector,
                query_filter=self._build_qdrant_filter(f),
                limit=fetch_limit,
                score_threshold=threshold,
                with_payload=True,
            )
        )

        hits = [self._parse_hit(h) for h in response.points]

        if f.min_price is not None or f.max_price is not None:
            hits = _filter_by_price(hits, f.min_price, f.max_price)

        if f.in_stock is not None:
            hits = _filter_by_stock(hits, f.in_stock)

        return [_to_source_document(h) for h in hits[:k]]

    async def search_for_context(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> tuple[list[SourceDocument], str]:
        """Return source documents **and** a pre-formatted LLM context string.

        The context string is a numbered list of products ready to be passed
        as ``extra_context`` to ``ChatOrchestrator._llm_reply``.  An empty
        string is returned when no products are found.

        Example context string::

            1. Samsung Galaxy A55 5G — 8.990.000 ₫ | Trả góp 0%, bảo hành 12 tháng
            2. Samsung Galaxy A35 5G — 7.490.000 ₫ | Ưu đãi đặc biệt tháng này
        """
        docs = await self.search(
            query,
            filters=filters,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        context = _format_context(docs) if docs else ""
        return docs, context

    # -------------------------------------------------------------------------
    # Qdrant filter builder
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_qdrant_filter(f: SearchFilters) -> Filter | None:
        """Translate *f* into a Qdrant ``Filter``.

        Only **brand** is handled at the Qdrant level (tokenised ``MatchText``
        on the ``title`` field).  Price and stock are post-filters because
        prices are stored as formatted strings and the stock field may be
        absent from older documents.
        """
        conditions: list[FieldCondition] = []

        if f.brand:
            conditions.append(
                FieldCondition(key="title", match=MatchText(text=f.brand))
            )

        return Filter(must=conditions) if conditions else None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Encode *text* with the SentenceTransformer model (runs synchronously)."""
        return self._model.encode(text, show_progress_bar=False).tolist()

    @staticmethod
    def _parse_hit(hit: object) -> _Hit:
        payload: dict = getattr(hit, "payload", None) or {}
        price_str: str = payload.get("current_price", "")
        return _Hit(
            point_id=str(hit.id),  # type: ignore[attr-defined]
            score=float(hit.score),  # type: ignore[attr-defined]
            title=payload.get("title", ""),
            current_price=price_str,
            price_vnd=_parse_price_vnd(price_str),
            url=payload.get("url", ""),
            product_promotion=payload.get("product_promotion", ""),
            product_specs=payload.get("product_specs", ""),
            color_options=payload.get("color_options", ""),
            in_stock=payload.get("in_stock"),
        )


# ---------------------------------------------------------------------------
# Pure module-level helpers
# ---------------------------------------------------------------------------


def _parse_price_vnd(price_str: str) -> float | None:
    """Parse a Vietnamese-formatted price string to a plain float (VND).

    Handles::

        "5.990.000 ₫"   → 5_990_000.0
        "15,990,000đ"   → 15_990_000.0
        "5990000"       → 5_990_000.0
        "Liên hệ"       → None
        ""              → None
    """
    if not price_str:
        return None
    digits = re.sub(r"[^\d]", "", price_str)
    return float(digits) if digits else None


def _filter_by_price(
    hits: list[_Hit],
    min_price: float | None,
    max_price: float | None,
) -> list[_Hit]:
    """Keep hits whose parsed VND price falls within [min_price, max_price]."""
    result = []
    for h in hits:
        if h.price_vnd is None:
            # Price is unparseable (e.g. "Liên hệ") — exclude when filtering.
            continue
        if min_price is not None and h.price_vnd < min_price:
            continue
        if max_price is not None and h.price_vnd > max_price:
            continue
        result.append(h)
    return result


def _filter_by_stock(hits: list[_Hit], in_stock: bool) -> list[_Hit]:
    """Keep only hits where the ``in_stock`` payload field equals *in_stock*.

    Hits whose payload lacks the field (``in_stock=None``) are excluded.
    """
    return [h for h in hits if h.in_stock is in_stock]


def _to_source_document(hit: _Hit) -> SourceDocument:
    """Convert a ``_Hit`` to the public ``SourceDocument`` response schema."""
    parts = [hit.current_price]
    if hit.product_specs:
        parts.append(hit.product_specs)
    if hit.color_options:
        parts.append(hit.color_options)
    if hit.in_stock:
        parts.append(hit.in_stock and "In Stock" or "Out of Stock")
    if hit.product_promotion:
        parts.append(hit.product_promotion[:200])
    snippet_str = " | ".join(p for p in parts if p)
    return SourceDocument(
        product_id=hit.point_id,
        product_name=hit.title,
        score=round(hit.score, 4),
        snippet=snippet_str or None,
    )


def _format_context(docs: list[SourceDocument]) -> str:
    """Render retrieved products as a numbered list for LLM prompt injection."""
    lines = []
    for i, doc in enumerate(docs, 1):
        line = f"{i}. {doc.product_name}"
        if doc.snippet:
            line += f" — {doc.snippet}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton factory for FastAPI dependency injection
# ---------------------------------------------------------------------------

_retriever_instance: ProductRetriever | None = None


def get_retriever(settings: Settings) -> ProductRetriever:
    """Return (or lazily create) the process-scoped ``ProductRetriever``.

    Intended use in route handlers::

        from app.dependencies import AppSettings
        from app.rag.retriever import get_retriever

        @router.post("/search")
        async def search(body: SearchRequest, settings: AppSettings):
            retriever = get_retriever(settings)
            docs, context = await retriever.search_for_context(body.query)
            ...

    The singleton is initialised on first call.  Because the embedding model
    is loaded at that point, the first request after a cold start will be
    slower than subsequent ones.  Warm it up during the FastAPI lifespan event
    if instant first-response latency matters.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ProductRetriever(settings)
    return _retriever_instance

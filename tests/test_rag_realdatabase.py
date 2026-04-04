"""Integration tests: vector search against real Qdrant cloud data.

Requirements
------------
- ``mobile_products`` collection ingested (run ``make ingest`` first).
- Valid ``QDRANT_URL`` and ``QDRANT_API_KEY`` in ``back-end/.env``.
- Internet access to Qdrant cloud and HuggingFace (first model download).

Run only these tests:
    pytest tests/test_rag_realdatabase.py -v

Exclude from a regular unit-test run:
    pytest tests/unit/ -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import TextIndexParams, TokenizerType

from app.config import Settings
from app.rag.retriever import ProductRetriever, SearchFilters, _parse_price_vnd

# Prevent SentenceTransformer from hitting HuggingFace on every instantiation.
# The model must already be cached locally (it is, after the first run).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Load back-end/.env before any Settings object is instantiated so that
# QDRANT_URL, QDRANT_API_KEY, etc. are present in the environment.
load_dotenv(Path(__file__).parents[1] / "back-end" / ".env", override=True)


# ---------------------------------------------------------------------------
# Module-scoped fixtures  (model loaded once → fast individual tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings() -> Settings:
    # Lower threshold for integration tests: the production value (0.6) is
    # intentionally strict and filters out most generic queries. 0.3 still
    # removes junk while allowing enough results to verify search correctness.
    return Settings(retrieval_score_threshold=0.3)


@pytest.fixture(scope="module", autouse=True)
def require_qdrant(settings: Settings) -> None:
    """Skip the whole module when Qdrant is unreachable or not yet ingested.

    Also ensures a full-text payload index exists on the ``title`` field so
    that ``MatchText`` brand filters work correctly.
    """
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10,
        )
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        pytest.skip(f"Qdrant cloud unreachable: {exc}")
        return  # unreachable, but satisfies type checker

    if settings.qdrant_collection not in existing:
        pytest.skip(
            f"Collection '{settings.qdrant_collection}' not found — "
            "run 'make ingest' to populate it first."
        )

    # Ensure a full-text index exists on `title` for MatchText brand filters.
    # This is idempotent — Qdrant ignores the call if the index already exists.
    try:
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="title",
            field_schema=TextIndexParams(type="text", tokenizer=TokenizerType.WORD),
        )
    except Exception:
        pass  # index may already exist or creation may not be supported


@pytest.fixture(scope="module")
def retriever(settings: Settings) -> ProductRetriever:
    """Real ProductRetriever — shared across all tests in this module.

    The sync QdrantClient has no event-loop affinity, so a single instance
    can safely be reused across tests running on different event loops.
    The SentenceTransformer model is loaded once (~2 s) and reused.
    HF_HUB_OFFLINE=1 (set at module top) prevents network calls on load.
    """
    return ProductRetriever(settings)


# ---------------------------------------------------------------------------
# 1. Basic connectivity and result shape
# ---------------------------------------------------------------------------


class TestBasicSearch:
    pytestmark = pytest.mark.asyncio

    async def test_generic_vietnamese_query_returns_results(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search("điện thoại giá rẻ")
        assert len(docs) > 0, "Expected results for a broad Vietnamese query"

    async def test_result_count_does_not_exceed_top_k(
        self, retriever: ProductRetriever, settings: Settings
    ):
        docs = await retriever.search("smartphone")
        assert len(docs) <= settings.retrieval_top_k

    async def test_all_scores_in_valid_range(self, retriever: ProductRetriever):
        docs = await retriever.search("điện thoại")
        assert all(0.0 <= d.score <= 1.0 for d in docs)

    async def test_results_ordered_by_descending_score(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search("điện thoại")
        scores = [d.score for d in docs]
        assert scores == sorted(scores, reverse=True), (
            "Results must come back best-first"
        )

    async def test_each_result_has_non_empty_id_and_name(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search("điện thoại")
        for doc in docs:
            assert doc.product_id, f"Empty product_id: {doc}"
            assert doc.product_name, f"Empty product_name: {doc}"

    async def test_top_k_one_returns_at_most_one_result(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search("iPhone", top_k=1)
        assert len(docs) <= 1

    async def test_top_k_two_returns_at_most_two_results(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search("Samsung Galaxy", top_k=2)
        assert len(docs) <= 2

    async def test_lower_threshold_returns_more_or_equal_results(
        self, retriever: ProductRetriever
    ):
        # settings uses 0.3; score_threshold=0.0 must yield ≥ as many results.
        default = await retriever.search("điện thoại")
        loose = await retriever.search("điện thoại", score_threshold=0.0)
        assert len(loose) >= len(default), (
            f"score_threshold=0.0 returned {len(loose)}, "
            f"default (0.3) returned {len(default)}"
        )

    async def test_higher_threshold_returns_fewer_or_equal_results(
        self, retriever: ProductRetriever
    ):
        default = await retriever.search("điện thoại")
        strict = await retriever.search("điện thoại", score_threshold=0.95)
        assert len(strict) <= len(default)

    async def test_iphone_and_samsung_queries_return_different_results(
        self, retriever: ProductRetriever
    ):
        iphone = {d.product_id for d in await retriever.search("iPhone mới nhất")}
        samsung = {d.product_id for d in await retriever.search("Samsung Galaxy")}
        assert iphone != samsung, (
            "Semantically different queries must not return identical result sets"
        )


# ---------------------------------------------------------------------------
# 2. Brand filter  (Qdrant-level MatchText on title field)
# ---------------------------------------------------------------------------


class TestBrandFilter:
    pytestmark = pytest.mark.asyncio

    async def test_samsung_filter_only_returns_samsung_products(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search(
            "điện thoại", filters=SearchFilters(brand="Samsung")
        )
        assert len(docs) > 0, "Expected Samsung products in the collection"
        for doc in docs:
            assert "samsung" in doc.product_name.lower(), (
                f"Non-Samsung result slipped through: {doc.product_name}"
            )

    async def test_iphone_filter_only_returns_iphone_products(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search(
            "điện thoại", filters=SearchFilters(brand="iPhone")
        )
        assert len(docs) > 0, "Expected iPhone products in the collection"
        for doc in docs:
            assert "iphone" in doc.product_name.lower(), (
                f"Non-iPhone result slipped through: {doc.product_name}"
            )

    async def test_brand_filter_narrows_result_set(
        self, retriever: ProductRetriever
    ):
        all_docs = await retriever.search("điện thoại")
        branded = await retriever.search(
            "điện thoại", filters=SearchFilters(brand="Samsung")
        )
        assert len(branded) <= len(all_docs)

    async def test_nonexistent_brand_returns_empty_list(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search(
            "điện thoại", filters=SearchFilters(brand="BrandXYZDoesNotExist99999")
        )
        assert docs == []


# ---------------------------------------------------------------------------
# 3. Price filter  (post-retrieval, parses Vietnamese price strings)
# ---------------------------------------------------------------------------


class TestPriceFilter:
    pytestmark = pytest.mark.asyncio

    async def test_max_price_10m_results_within_range(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search(
            "điện thoại", filters=SearchFilters(max_price=10_000_000)
        )
        for doc in docs:
            if doc.snippet:
                price = _parse_price_vnd(doc.snippet.split(" | ")[0])
                if price is not None:
                    assert price <= 10_000_000, (
                        f"{doc.product_name}: {price:,.0f} ₫ exceeds max 10,000,000 ₫"
                    )

    async def test_min_price_15m_results_within_range(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search(
            "điện thoại cao cấp", filters=SearchFilters(min_price=15_000_000)
        )
        for doc in docs:
            if doc.snippet:
                price = _parse_price_vnd(doc.snippet.split(" | ")[0])
                if price is not None:
                    assert price >= 15_000_000, (
                        f"{doc.product_name}: {price:,.0f} ₫ below min 15,000,000 ₫"
                    )

    async def test_price_range_5m_to_15m(self, retriever: ProductRetriever):
        docs = await retriever.search(
            "điện thoại",
            filters=SearchFilters(min_price=5_000_000, max_price=15_000_000),
        )
        for doc in docs:
            if doc.snippet:
                price = _parse_price_vnd(doc.snippet.split(" | ")[0])
                if price is not None:
                    assert 5_000_000 <= price <= 15_000_000, (
                        f"{doc.product_name}: {price:,.0f} ₫ outside [5M, 15M]"
                    )

    async def test_impossible_price_range_returns_empty(
        self, retriever: ProductRetriever
    ):
        # min > max — no product can satisfy this.
        docs = await retriever.search(
            "điện thoại",
            filters=SearchFilters(min_price=50_000_000, max_price=1_000_000),
        )
        assert docs == []


# ---------------------------------------------------------------------------
# 4. Combined brand + price filters
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    pytestmark = pytest.mark.asyncio

    async def test_samsung_under_10m_all_correct_brand_and_price(
        self, retriever: ProductRetriever
    ):
        docs = await retriever.search(
            "điện thoại Samsung",
            filters=SearchFilters(brand="Samsung", max_price=10_000_000),
        )
        for doc in docs:
            assert "samsung" in doc.product_name.lower()
            if doc.snippet:
                price = _parse_price_vnd(doc.snippet.split(" | ")[0])
                if price is not None:
                    assert price <= 10_000_000

    async def test_combined_filters_subset_of_brand_only(
        self, retriever: ProductRetriever
    ):
        brand_only = await retriever.search(
            "điện thoại", filters=SearchFilters(brand="Samsung")
        )
        brand_and_price = await retriever.search(
            "điện thoại",
            filters=SearchFilters(brand="Samsung", max_price=10_000_000),
        )
        assert len(brand_and_price) <= len(brand_only), (
            "Adding a price filter must never increase the result count"
        )


# ---------------------------------------------------------------------------
# 5. search_for_context
# ---------------------------------------------------------------------------


class TestSearchForContext:
    pytestmark = pytest.mark.asyncio

    async def test_returns_docs_and_non_empty_context(
        self, retriever: ProductRetriever
    ):
        docs, ctx = await retriever.search_for_context("điện thoại Samsung")
        assert len(docs) > 0
        assert ctx != ""

    async def test_context_is_numbered_list(self, retriever: ProductRetriever):
        _, ctx = await retriever.search_for_context("điện thoại")
        lines = ctx.splitlines()
        assert lines[0].startswith("1.")
        if len(lines) > 1:
            assert lines[1].startswith("2.")

    async def test_line_count_matches_doc_count(self, retriever: ProductRetriever):
        docs, ctx = await retriever.search_for_context("điện thoại")
        assert len(ctx.splitlines()) == len(docs)

    async def test_all_product_names_appear_in_context(
        self, retriever: ProductRetriever
    ):
        docs, ctx = await retriever.search_for_context("Samsung Galaxy")
        for doc in docs:
            assert doc.product_name in ctx

    async def test_no_results_gives_empty_context(
        self, retriever: ProductRetriever
    ):
        docs, ctx = await retriever.search_for_context(
            "điện thoại", filters=SearchFilters(brand="BrandXYZDoesNotExist99999")
        )
        assert docs == []
        assert ctx == ""

    async def test_context_is_plain_string_safe_for_llm(
        self, retriever: ProductRetriever
    ):
        """Context must be injectable into an LLM prompt without modification."""
        _, ctx = await retriever.search_for_context("điện thoại tầm trung")
        assert isinstance(ctx, str)
        assert "\x00" not in ctx

"""Unit tests for app.rag.retriever.

All Qdrant and SentenceTransformer calls are mocked so these tests run
without any external services or network access.

Test groups
-----------
TestParsePriceVnd          pure function — Vietnamese price string parsing
TestFilterByPrice          pure function — numeric price range post-filter
TestFilterByStock          pure function — in_stock boolean post-filter
TestToSourceDocument       pure function — _Hit → SourceDocument conversion
TestFormatContext          pure function — numbered list rendering
TestBuildQdrantFilter      static method — Qdrant Filter construction
TestParseHit               static method — raw Qdrant hit → _Hit
TestProductRetrieverSearch async method  — full search pipeline (mocked I/O)
TestSearchForContext        async method  — docs + context string wrapper
TestGetRetriever           factory       — process-scoped singleton behaviour
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http.models import FieldCondition, Filter, MatchText

import app.rag.retriever as retriever_module
from app.api.schemas.chat import SourceDocument
from app.config import Settings
from app.rag.retriever import (
    ProductRetriever,
    SearchFilters,
    _Hit,
    _filter_by_price,
    _filter_by_stock,
    _format_context,
    _parse_price_vnd,
    _to_source_document,
    get_retriever,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> Settings:
    """Minimal Settings with dummy secrets suitable for unit tests."""
    return Settings(
        anthropic_api_key="sk-test-key",
        secret_key="test-secret-32-chars-xxxxxxxxxxx",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="test_collection",
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        retrieval_top_k=3,
        retrieval_score_threshold=0.5,
    )


@pytest.fixture()
def mock_encode_result() -> MagicMock:
    """Fake numpy-array-like returned by SentenceTransformer.encode."""
    arr = MagicMock()
    arr.tolist.return_value = [0.0] * 384
    return arr


@pytest.fixture()
def retriever(settings: Settings, mock_encode_result: MagicMock) -> ProductRetriever:
    """ProductRetriever with both Qdrant client and embedding model mocked.

    The mock objects are attached as ``_mock_qdrant`` and ``_mock_model``
    for inspection inside individual tests.
    """
    with (
        patch("app.rag.retriever.QdrantClient") as mock_qdrant_cls,
        patch("sentence_transformers.SentenceTransformer") as mock_st_cls,
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_encode_result
        mock_st_cls.return_value = mock_model

        # Sync client — query_points is a regular (not async) method.
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        r = ProductRetriever(settings)
        # Attach mocks so test methods can assert on call args.
        r._mock_qdrant = mock_qdrant  # type: ignore[attr-defined]
        r._mock_model = mock_model    # type: ignore[attr-defined]
        return r


@pytest.fixture()
def reset_singleton():
    """Reset the process-scoped singleton before and after each test."""
    retriever_module._retriever_instance = None
    yield
    retriever_module._retriever_instance = None


# ---------------------------------------------------------------------------
# Builder helper — creates a mock Qdrant ScoredPoint
# ---------------------------------------------------------------------------


def _qdrant_hit(
    point_id: str = "abc-123",
    score: float = 0.85,
    title: str = "Samsung Galaxy S24",
    current_price: str = "19.990.000 ₫",
    product_promotion: str = "Giảm 10% tháng này",
    product_specs: str = "RAM 8GB, 256GB",
    color_options: str = "Đen, Trắng",
    url: str = "https://shop.example.com/product",
    in_stock: bool | None = None,
) -> MagicMock:
    """Return a mock object shaped like a Qdrant ``ScoredPoint``."""
    hit = MagicMock()
    hit.id = point_id
    hit.score = score
    payload: dict = {
        "title": title,
        "current_price": current_price,
        "product_promotion": product_promotion,
        "product_specs": product_specs,
        "color_options": color_options,
        "url": url,
    }
    if in_stock is not None:
        payload["in_stock"] = in_stock
    hit.payload = payload
    return hit


def _make_hit(
    price: str = "5.000.000 ₫",
    in_stock: bool | None = None,
    point_id: str = "id",
    score: float = 0.9,
) -> _Hit:
    """Construct a ``_Hit`` dataclass directly for post-filter tests."""
    return _Hit(
        point_id=point_id,
        score=score,
        title="Test Phone",
        current_price=price,
        price_vnd=_parse_price_vnd(price),
        url="",
        product_promotion="",
        product_specs="",
        color_options="",
        in_stock=in_stock,
    )


# ===========================================================================
# 1. _parse_price_vnd
# ===========================================================================


class TestParsePriceVnd:
    def test_dot_separated_with_symbol(self):
        assert _parse_price_vnd("5.990.000 ₫") == 5_990_000.0

    def test_comma_separated_with_dong(self):
        assert _parse_price_vnd("15,990,000đ") == 15_990_000.0

    def test_plain_integer_string(self):
        assert _parse_price_vnd("5990000") == 5_990_000.0

    def test_mixed_dot_and_dong(self):
        assert _parse_price_vnd("28.990.000đ") == 28_990_000.0

    def test_high_price(self):
        assert _parse_price_vnd("99.999.999 ₫") == 99_999_999.0

    def test_zero_price(self):
        assert _parse_price_vnd("0đ") == 0.0

    def test_contact_price_returns_none(self):
        assert _parse_price_vnd("Liên hệ") is None

    def test_empty_string_returns_none(self):
        assert _parse_price_vnd("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_price_vnd("   ") is None

    def test_non_numeric_text_returns_none(self):
        assert _parse_price_vnd("Thương lượng") is None


# ===========================================================================
# 2. _filter_by_price
# ===========================================================================


class TestFilterByPrice:
    def _hits(self, prices: list[str]) -> list[_Hit]:
        return [_make_hit(price=p, point_id=str(i)) for i, p in enumerate(prices)]

    def test_min_price_removes_cheaper(self):
        result = _filter_by_price(
            self._hits(["3.000.000 ₫", "7.000.000 ₫", "12.000.000 ₫"]),
            min_price=5_000_000,
            max_price=None,
        )
        assert len(result) == 2
        assert result[0].price_vnd == 7_000_000.0
        assert result[1].price_vnd == 12_000_000.0

    def test_max_price_removes_dearer(self):
        result = _filter_by_price(
            self._hits(["3.000.000 ₫", "7.000.000 ₫", "12.000.000 ₫"]),
            min_price=None,
            max_price=10_000_000,
        )
        assert len(result) == 2

    def test_range_keeps_middle_only(self):
        result = _filter_by_price(
            self._hits(["3.000.000 ₫", "7.000.000 ₫", "12.000.000 ₫"]),
            min_price=5_000_000,
            max_price=10_000_000,
        )
        assert len(result) == 1
        assert result[0].price_vnd == 7_000_000.0

    def test_boundaries_are_inclusive(self):
        result = _filter_by_price(
            self._hits(["5.000.000 ₫", "10.000.000 ₫"]),
            min_price=5_000_000,
            max_price=10_000_000,
        )
        assert len(result) == 2

    def test_unparseable_price_excluded_when_filter_active(self):
        result = _filter_by_price(
            self._hits(["Liên hệ", "7.000.000 ₫"]),
            min_price=1_000_000,
            max_price=None,
        )
        assert len(result) == 1
        assert result[0].price_vnd == 7_000_000.0

    def test_no_bounds_still_excludes_none_prices(self):
        # The function is only called when at least one bound is set,
        # but defensively it still drops None prices.
        result = _filter_by_price(
            self._hits(["3.000.000 ₫", "Liên hệ", "9.000.000 ₫"]),
            min_price=None,
            max_price=None,
        )
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        assert _filter_by_price([], 0, 100_000_000) == []

    def test_all_filtered_out_returns_empty(self):
        result = _filter_by_price(
            self._hits(["5.000.000 ₫", "8.000.000 ₫"]),
            min_price=20_000_000,
            max_price=None,
        )
        assert result == []


# ===========================================================================
# 3. _filter_by_stock
# ===========================================================================


class TestFilterByStock:
    def test_in_stock_true_keeps_only_true(self):
        hits = [
            _make_hit(in_stock=True, point_id="a"),
            _make_hit(in_stock=False, point_id="b"),
            _make_hit(in_stock=None, point_id="c"),
        ]
        result = _filter_by_stock(hits, in_stock=True)
        assert len(result) == 1
        assert result[0].point_id == "a"

    def test_in_stock_false_keeps_only_false(self):
        hits = [
            _make_hit(in_stock=True, point_id="a"),
            _make_hit(in_stock=False, point_id="b"),
            _make_hit(in_stock=None, point_id="c"),
        ]
        result = _filter_by_stock(hits, in_stock=False)
        assert len(result) == 1
        assert result[0].point_id == "b"

    def test_missing_field_excluded_when_true_requested(self):
        hits = [_make_hit(in_stock=None), _make_hit(in_stock=None)]
        assert _filter_by_stock(hits, in_stock=True) == []

    def test_empty_input_returns_empty(self):
        assert _filter_by_stock([], in_stock=True) == []

    def test_all_in_stock_all_returned(self):
        hits = [_make_hit(in_stock=True, point_id=str(i)) for i in range(4)]
        assert len(_filter_by_stock(hits, in_stock=True)) == 4


# ===========================================================================
# 4. _to_source_document
# ===========================================================================


class TestToSourceDocument:
    def _hit(self, **kwargs) -> _Hit:
        defaults = dict(
            point_id="uid-1",
            score=0.8765,
            title="iPhone 15 Pro",
            current_price="27.990.000 ₫",
            price_vnd=27_990_000.0,
            url="https://example.com/iphone-15-pro",
            product_promotion="Trả góp 0%",
            product_specs="A17 Pro chip",
            color_options="Titan Đen",
            in_stock=True,
        )
        defaults.update(kwargs)
        return _Hit(**defaults)

    def test_product_id_maps_to_point_id(self):
        doc = _to_source_document(self._hit(point_id="my-id"))
        assert doc.product_id == "my-id"

    def test_product_name_maps_to_title(self):
        doc = _to_source_document(self._hit(title="Xiaomi 14T Pro"))
        assert doc.product_name == "Xiaomi 14T Pro"

    def test_score_rounded_to_4_decimal_places(self):
        doc = _to_source_document(self._hit(score=0.876543))
        assert doc.score == 0.8765

    def test_snippet_contains_price(self):
        doc = _to_source_document(self._hit(current_price="27.990.000 ₫", product_promotion=""))
        assert "27.990.000 ₫" in doc.snippet

    def test_snippet_contains_promotion(self):
        doc = _to_source_document(self._hit(product_promotion="Giảm 2 triệu đồng"))
        assert "Giảm 2 triệu đồng" in doc.snippet

    def test_snippet_is_none_when_both_price_and_promo_empty(self):
        doc = _to_source_document(self._hit(current_price="", product_promotion=""))
        assert doc.snippet is None

    def test_promotion_truncated_at_200_chars(self):
        long_promo = "X" * 300
        doc = _to_source_document(self._hit(current_price="", product_promotion=long_promo))
        # Only the first 200 chars of promotion are included.
        assert len(doc.snippet) == 200

    def test_returns_source_document_instance(self):
        assert isinstance(_to_source_document(self._hit()), SourceDocument)

    def test_score_within_valid_range(self):
        doc = _to_source_document(self._hit(score=0.5))
        assert 0.0 <= doc.score <= 1.0


# ===========================================================================
# 5. _format_context
# ===========================================================================


class TestFormatContext:
    def _doc(self, name: str, snippet: str | None = None) -> SourceDocument:
        return SourceDocument(product_id="id", product_name=name, score=0.9, snippet=snippet)

    def test_empty_list_returns_empty_string(self):
        assert _format_context([]) == ""

    def test_single_doc_no_snippet(self):
        assert _format_context([self._doc("Xiaomi 14T")]) == "1. Xiaomi 14T"

    def test_numbering_starts_at_one(self):
        docs = [self._doc(f"Phone {i}") for i in range(3)]
        lines = _format_context(docs).splitlines()
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")
        assert lines[2].startswith("3.")

    def test_snippet_appended_after_dash(self):
        doc = self._doc("iPhone 15", snippet="28.990.000 ₫ | Trả góp 0%")
        ctx = _format_context([doc])
        assert "— 28.990.000 ₫ | Trả góp 0%" in ctx

    def test_no_snippet_means_no_dash(self):
        ctx = _format_context([self._doc("iPhone 15", snippet=None)])
        assert "—" not in ctx

    def test_each_product_on_its_own_line(self):
        docs = [self._doc(f"Phone {i}") for i in range(4)]
        assert len(_format_context(docs).splitlines()) == 4

    def test_product_name_present_in_output(self):
        ctx = _format_context([self._doc("Samsung Galaxy S24 Ultra")])
        assert "Samsung Galaxy S24 Ultra" in ctx


# ===========================================================================
# 6. ProductRetriever._build_qdrant_filter
# ===========================================================================


class TestBuildQdrantFilter:
    def test_empty_filters_returns_none(self):
        assert ProductRetriever._build_qdrant_filter(SearchFilters()) is None

    def test_brand_creates_match_text_on_title(self):
        result = ProductRetriever._build_qdrant_filter(SearchFilters(brand="Samsung"))
        assert isinstance(result, Filter)
        assert len(result.must) == 1
        cond: FieldCondition = result.must[0]
        assert cond.key == "title"
        assert isinstance(cond.match, MatchText)
        assert cond.match.text == "Samsung"

    def test_price_only_returns_none(self):
        # Price is a post-filter; nothing is sent to Qdrant for it.
        f = SearchFilters(min_price=5_000_000, max_price=15_000_000)
        assert ProductRetriever._build_qdrant_filter(f) is None

    def test_stock_only_returns_none(self):
        # in_stock is a post-filter; nothing is sent to Qdrant for it.
        assert ProductRetriever._build_qdrant_filter(SearchFilters(in_stock=True)) is None

    def test_brand_plus_price_only_brand_in_filter(self):
        f = SearchFilters(brand="iPhone", min_price=20_000_000)
        result = ProductRetriever._build_qdrant_filter(f)
        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "title"

    def test_empty_brand_string_returns_none(self):
        assert ProductRetriever._build_qdrant_filter(SearchFilters(brand="")) is None

    def test_returns_filter_type(self):
        result = ProductRetriever._build_qdrant_filter(SearchFilters(brand="OPPO"))
        assert isinstance(result, Filter)


# ===========================================================================
# 7. ProductRetriever._parse_hit
# ===========================================================================


class TestParseHit:
    def test_basic_fields_extracted_correctly(self):
        raw = _qdrant_hit(point_id="abc-123", score=0.91, title="OPPO Reno 12",
                          current_price="9.990.000 ₫")
        hit = ProductRetriever._parse_hit(raw)
        assert hit.point_id == "abc-123"
        assert hit.score == pytest.approx(0.91)
        assert hit.title == "OPPO Reno 12"
        assert hit.current_price == "9.990.000 ₫"
        assert hit.price_vnd == 9_990_000.0

    def test_in_stock_true_propagated(self):
        hit = ProductRetriever._parse_hit(_qdrant_hit(in_stock=True))
        assert hit.in_stock is True

    def test_in_stock_false_propagated(self):
        hit = ProductRetriever._parse_hit(_qdrant_hit(in_stock=False))
        assert hit.in_stock is False

    def test_in_stock_absent_is_none(self):
        hit = ProductRetriever._parse_hit(_qdrant_hit())
        assert hit.in_stock is None

    def test_empty_payload_defaults_to_empty_strings(self):
        raw = MagicMock()
        raw.id = "id-1"
        raw.score = 0.8
        raw.payload = {}
        hit = ProductRetriever._parse_hit(raw)
        assert hit.title == ""
        assert hit.current_price == ""
        assert hit.price_vnd is None

    def test_none_payload_handled_gracefully(self):
        raw = MagicMock()
        raw.id = "id-2"
        raw.score = 0.7
        raw.payload = None
        hit = ProductRetriever._parse_hit(raw)
        assert hit.title == ""
        assert hit.in_stock is None

    def test_unparseable_price_gives_none_price_vnd(self):
        raw = _qdrant_hit(current_price="Liên hệ")
        hit = ProductRetriever._parse_hit(raw)
        assert hit.price_vnd is None

    def test_returns_hit_dataclass(self):
        assert isinstance(ProductRetriever._parse_hit(_qdrant_hit()), _Hit)


# ===========================================================================
# 8. ProductRetriever.search  (async, fully mocked)
# ===========================================================================


class TestProductRetrieverSearch:
    pytestmark = pytest.mark.asyncio

    async def test_returns_list_of_source_documents(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[  # type: ignore[attr-defined]
            _qdrant_hit(point_id="1", score=0.95, title="Samsung Galaxy A55"),
            _qdrant_hit(point_id="2", score=0.80, title="Samsung Galaxy A35"),
        ])
        docs = await retriever.search("điện thoại Samsung")
        assert len(docs) == 2
        assert all(isinstance(d, SourceDocument) for d in docs)

    def _qdrant_return(self, retriever: ProductRetriever, hits: list) -> None:
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=hits)  # type: ignore[attr-defined]

    async def test_preserves_result_order(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id="1", score=0.95, title="Top Result"),
            _qdrant_hit(point_id="2", score=0.75, title="Second Result"),
        ])
        docs = await retriever.search("query")
        assert docs[0].product_name == "Top Result"
        assert docs[1].product_name == "Second Result"

    async def test_empty_qdrant_response_returns_empty_list(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        assert await retriever.search("no match") == []

    async def test_top_k_trims_results(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id=str(i), score=0.9 - i * 0.05) for i in range(5)
        ])
        docs = await retriever.search("query", top_k=2)
        assert len(docs) == 2

    async def test_default_top_k_from_settings(self, retriever: ProductRetriever):
        # settings.retrieval_top_k = 3; Qdrant returns exactly 3 hits.
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id=str(i)) for i in range(3)
        ])
        docs = await retriever.search("query")
        assert len(docs) == 3

    async def test_score_threshold_forwarded_to_qdrant(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query", score_threshold=0.75)
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["score_threshold"] == 0.75

    async def test_collection_name_forwarded_to_qdrant(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query")
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["collection_name"] == "test_collection"

    async def test_with_payload_true_always_set(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query")
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["with_payload"] is True

    async def test_brand_filter_reaches_qdrant(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query", filters=SearchFilters(brand="iPhone"))
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        q_filter: Filter = kwargs["query_filter"]
        assert q_filter is not None
        assert q_filter.must[0].key == "title"
        assert q_filter.must[0].match.text == "iPhone"

    async def test_no_brand_passes_none_filter_to_qdrant(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query")
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["query_filter"] is None

    async def test_over_fetch_when_price_filter_active(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query", top_k=3, filters=SearchFilters(max_price=10_000_000))
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["limit"] == 15  # 3 * 5

    async def test_over_fetch_when_stock_filter_active(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query", top_k=2, filters=SearchFilters(in_stock=True))
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["limit"] == 10  # 2 * 5

    async def test_no_over_fetch_without_post_filters(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("query", top_k=3)
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["limit"] == 3

    async def test_price_range_post_filter_applied(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id="cheap",     score=0.90, current_price="4.000.000 ₫"),
            _qdrant_hit(point_id="mid",       score=0.85, current_price="8.000.000 ₫"),
            _qdrant_hit(point_id="expensive", score=0.80, current_price="20.000.000 ₫"),
        ])
        docs = await retriever.search(
            "query",
            filters=SearchFilters(min_price=5_000_000, max_price=15_000_000),
        )
        ids = {d.product_id for d in docs}
        assert "mid" in ids
        assert "cheap" not in ids
        assert "expensive" not in ids

    async def test_max_price_filter_excludes_above(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id="ok",       score=0.9, current_price="9.000.000 ₫"),
            _qdrant_hit(point_id="too_dear", score=0.8, current_price="25.000.000 ₫"),
        ])
        docs = await retriever.search("query", filters=SearchFilters(max_price=10_000_000))
        assert len(docs) == 1
        assert docs[0].product_id == "ok"

    async def test_stock_filter_keeps_only_in_stock(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id="yes",     score=0.90, in_stock=True),
            _qdrant_hit(point_id="no",      score=0.85, in_stock=False),
            _qdrant_hit(point_id="unknown", score=0.80),  # no in_stock field
        ])
        docs = await retriever.search("query", filters=SearchFilters(in_stock=True))
        assert len(docs) == 1
        assert docs[0].product_id == "yes"

    async def test_stock_false_keeps_only_out_of_stock(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id="yes",  score=0.90, in_stock=True),
            _qdrant_hit(point_id="sold", score=0.85, in_stock=False),
        ])
        docs = await retriever.search("query", filters=SearchFilters(in_stock=False))
        assert len(docs) == 1
        assert docs[0].product_id == "sold"

    async def test_embedding_called_with_exact_query(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [])
        await retriever.search("tìm điện thoại giá rẻ")
        retriever._mock_model.encode.assert_called_once_with(  # type: ignore[attr-defined]
            "tìm điện thoại giá rẻ", show_progress_bar=False
        )

    async def test_combined_brand_and_price_filters(self, retriever: ProductRetriever):
        self._qdrant_return(retriever, [
            _qdrant_hit(point_id="s1", score=0.95, title="Samsung A55",
                        current_price="8.990.000 ₫"),
            _qdrant_hit(point_id="s2", score=0.90, title="Samsung S24",
                        current_price="22.990.000 ₫"),
        ])
        docs = await retriever.search(
            "Samsung giá dưới 10 triệu",
            filters=SearchFilters(brand="Samsung", max_price=10_000_000),
        )
        assert len(docs) == 1
        assert docs[0].product_id == "s1"


# ===========================================================================
# 9. ProductRetriever.search_for_context
# ===========================================================================


class TestSearchForContext:
    pytestmark = pytest.mark.asyncio

    async def test_returns_tuple_of_docs_and_string(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[  # type: ignore[attr-defined]
            _qdrant_hit(point_id="1", score=0.9, title="Samsung Galaxy S24",
                        current_price="19.990.000 ₫", product_promotion="Giảm 2 triệu")
        ])
        docs, ctx = await retriever.search_for_context("Samsung")
        assert isinstance(docs, list)
        assert isinstance(ctx, str)

    async def test_context_contains_product_name(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[  # type: ignore[attr-defined]
            _qdrant_hit(title="Xiaomi 14T Pro", current_price="12.990.000 ₫")
        ])
        _, ctx = await retriever.search_for_context("query")
        assert "Xiaomi 14T Pro" in ctx

    async def test_context_starts_with_number(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[  # type: ignore[attr-defined]
            _qdrant_hit(title="iPhone 15")
        ])
        _, ctx = await retriever.search_for_context("iPhone")
        assert ctx.startswith("1.")

    async def test_empty_results_return_empty_context(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[])  # type: ignore[attr-defined]
        docs, ctx = await retriever.search_for_context("nothing")
        assert docs == []
        assert ctx == ""

    async def test_one_line_per_product(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[  # type: ignore[attr-defined]
            _qdrant_hit(point_id=str(i), title=f"Phone {i}") for i in range(3)
        ])
        _, ctx = await retriever.search_for_context("query")
        assert len(ctx.splitlines()) == 3

    async def test_docs_and_context_are_consistent(self, retriever: ProductRetriever):
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[  # type: ignore[attr-defined]
            _qdrant_hit(title="OPPO Reno 12", current_price="9.990.000 ₫")
        ])
        docs, ctx = await retriever.search_for_context("OPPO")
        assert len(docs) == 1
        assert docs[0].product_name in ctx

    async def test_filters_forwarded_to_search(self, retriever: ProductRetriever):
        """Filters passed to search_for_context reach the underlying Qdrant call."""
        retriever._mock_qdrant.query_points.return_value = MagicMock(points=[])  # type: ignore[attr-defined]
        await retriever.search_for_context(
            "query", filters=SearchFilters(brand="Apple")
        )
        kwargs = retriever._mock_qdrant.query_points.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["query_filter"] is not None
        assert kwargs["query_filter"].must[0].match.text == "Apple"


# ===========================================================================
# 10. get_retriever — singleton behaviour
# ===========================================================================


class TestGetRetriever:
    def test_returns_product_retriever_instance(
        self, settings: Settings, reset_singleton
    ):
        with (
            patch("app.rag.retriever.QdrantClient"),
            patch("sentence_transformers.SentenceTransformer"),
        ):
            r = get_retriever(settings)
        assert isinstance(r, ProductRetriever)

    def test_repeated_calls_return_same_object(
        self, settings: Settings, reset_singleton
    ):
        with (
            patch("app.rag.retriever.QdrantClient"),
            patch("sentence_transformers.SentenceTransformer"),
        ):
            r1 = get_retriever(settings)
            r2 = get_retriever(settings)
        assert r1 is r2

    def test_model_instantiated_only_once(
        self, settings: Settings, reset_singleton
    ):
        with (
            patch("app.rag.retriever.QdrantClient"),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):
            get_retriever(settings)
            get_retriever(settings)
            get_retriever(settings)
        assert mock_st.call_count == 1

    def test_qdrant_client_instantiated_only_once(
        self, settings: Settings, reset_singleton
    ):
        with (
            patch("app.rag.retriever.QdrantClient") as mock_qd,
            patch("sentence_transformers.SentenceTransformer"),
        ):
            get_retriever(settings)
            get_retriever(settings)
        assert mock_qd.call_count == 1

    def test_reset_allows_new_instance(self, settings: Settings, reset_singleton):
        with (
            patch("app.rag.retriever.QdrantClient"),
            patch("sentence_transformers.SentenceTransformer"),
        ):
            r1 = get_retriever(settings)
            retriever_module._retriever_instance = None
            r2 = get_retriever(settings)
        assert r1 is not r2

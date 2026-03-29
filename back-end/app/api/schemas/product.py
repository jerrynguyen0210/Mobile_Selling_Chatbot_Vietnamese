"""Pydantic schemas for product endpoints."""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"


class SortField(StrEnum):
    PRICE = "price"
    RELEVANCE = "relevance"
    RATING = "rating"
    NAME = "name"


# ---------------------------------------------------------------------------
# Search filters
# ---------------------------------------------------------------------------


class PriceRange(BaseModel):
    min: Decimal | None = Field(default=None, ge=0)
    max: Decimal | None = Field(default=None, ge=0)


class SearchFilters(BaseModel):
    """Query parameters / body for product search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Natural-language or keyword search query (Vietnamese supported)",
    )
    brands: list[str] = Field(
        default_factory=list,
        description="Filter by brand names, e.g. ['Samsung', 'Apple']",
    )
    price_range: PriceRange = Field(default_factory=PriceRange)
    ram_gb: list[int] = Field(default_factory=list, description="Filter by RAM options in GB")
    storage_gb: list[int] = Field(default_factory=list, description="Filter by storage options in GB")
    os: list[str] = Field(default_factory=list, description="Operating systems, e.g. ['Android', 'iOS']")
    min_rating: float | None = Field(default=None, ge=0.0, le=5.0)
    in_stock_only: bool = Field(default=False)
    sort_by: SortField = Field(default=SortField.RELEVANCE)
    sort_order: SortOrder = Field(default=SortOrder.DESC)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


# ---------------------------------------------------------------------------
# Product card (list / search result)
# ---------------------------------------------------------------------------


class ProductSpec(BaseModel):
    """Key technical specifications shown in a product card."""

    ram_gb: int | None = None
    storage_gb: int | None = None
    display_inches: float | None = None
    battery_mah: int | None = None
    camera_mp: int | None = None
    os: str | None = None
    chipset: str | None = None


class ProductCard(BaseModel):
    """Lightweight product representation used in search results."""

    product_id: str
    name: str
    brand: str
    price: Decimal
    currency: str = "VND"
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    review_count: int = Field(default=0, ge=0)
    image_url: HttpUrl | None = None
    in_stock: bool = True
    specs: ProductSpec = Field(default_factory=ProductSpec)
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score from vector search",
    )


class ProductSearchResponse(BaseModel):
    """Paginated search results."""

    results: list[ProductCard]
    total: int
    page: int
    page_size: int
    query: str


# ---------------------------------------------------------------------------
# Product detail
# ---------------------------------------------------------------------------


class ProductDetail(ProductCard):
    """Full product information including description and all specs."""

    description: str | None = None
    full_specs: dict[str, str | int | float | bool] = Field(default_factory=dict)
    images: list[HttpUrl] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Product comparison
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    product_ids: list[str] = Field(..., min_length=2, max_length=5)


class CompareRow(BaseModel):
    """A single spec row in the comparison table."""

    spec_name: str
    values: dict[str, str]  # product_id -> formatted value


class ProductCompare(BaseModel):
    """Side-by-side comparison of multiple products."""

    products: list[ProductCard]
    comparison_rows: list[CompareRow]

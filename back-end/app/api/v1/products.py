"""Product endpoints: semantic search, detail retrieval, comparison."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.product import (
    CompareRequest,
    ProductCompare,
    ProductDetail,
    ProductSearchResponse,
    SearchFilters,
)
from app.dependencies import AppSettings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["products"])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=ProductSearchResponse,
    summary="Semantic / keyword search over the product catalogue",
)
async def search_products(
    filters: SearchFilters,
    settings: AppSettings,
) -> ProductSearchResponse:
    """
    Embeds the query text, queries Qdrant for the nearest product vectors,
    then applies scalar filters (brand, price, RAM, OS, …) in-memory before
    returning a paginated result set.
    """
    # TODO: wire up EmbeddingService + QdrantService + scalar post-filtering
    logger.info(
        "Product search: query=%r page=%d page_size=%d",
        filters.query,
        filters.page,
        filters.page_size,
    )

    return ProductSearchResponse(
        results=[],
        total=0,
        page=filters.page,
        page_size=filters.page_size,
        query=filters.query,
    )


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


@router.get(
    "/{product_id}",
    response_model=ProductDetail,
    summary="Retrieve full product details by ID",
)
async def get_product(
    product_id: str,
) -> ProductDetail:
    """
    Fetches complete product information including full specs, description,
    and image gallery from the database.
    """
    # TODO: replace stub with real DB query
    logger.info("Product detail requested: %s", product_id)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Product '{product_id}' not found.",
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=ProductCompare,
    summary="Side-by-side comparison of 2–5 products",
)
async def compare_products(
    body: CompareRequest,
) -> ProductCompare:
    """
    Fetches the requested products and builds a structured comparison table
    aligning specs side-by-side so users can make an informed purchase decision.
    """
    # TODO: fetch products from DB and build comparison rows
    logger.info("Compare requested for products: %s", body.product_ids)

    missing = body.product_ids  # placeholder — treat all as missing until wired
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Products not found: {missing}",
        )

    return ProductCompare(products=[], comparison_rows=[])

"""Webhook endpoints for external system notifications."""

import hashlib
import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from pydantic import BaseModel

from app.dependencies import AppSettings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhooks"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ProductUpdatePayload(BaseModel):
    """Payload sent by the catalogue system when products change."""

    event: str  # "created" | "updated" | "deleted"
    product_ids: list[str]
    source: str | None = None


class WebhookResponse(BaseModel):
    accepted: bool
    queued_products: int


# ---------------------------------------------------------------------------
# Signature verification helper
# ---------------------------------------------------------------------------


def _verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 webhook signature sent in ``X-Webhook-Signature``."""
    expected = hmac.new(
        secret.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def _reindex_products(product_ids: list[str], settings: AppSettings) -> None:
    """Re-embed and upsert the given products into Qdrant."""
    logger.info("Re-indexing %d product(s): %s", len(product_ids), product_ids)
    # TODO: load products from DB → generate embeddings → upsert to Qdrant
    # from app.services.ingestion import IngestionService
    # await IngestionService(settings).reindex(product_ids)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/product-update",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger re-indexing when the product catalogue changes",
)
async def product_update_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: AppSettings,
    x_webhook_signature: str | None = Header(default=None, alias="X-Webhook-Signature"),
) -> WebhookResponse:
    """
    Receives a product-update event from the catalogue management system and
    enqueues a background re-indexing job so the Qdrant vector store stays in
    sync with the latest product data.

    Requests must include a valid ``X-Webhook-Signature: sha256=<hex>`` header
    computed with the ``SECRET_KEY`` configured on this server.
    """
    raw_body = await request.body()

    # Signature verification (skip in development when no signature is provided)
    if x_webhook_signature:
        if not _verify_signature(raw_body, x_webhook_signature, settings.secret_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature.",
            )
    else:
        logger.warning(
            "Webhook received without X-Webhook-Signature — skipping verification"
        )

    import json

    try:
        payload = ProductUpdatePayload(**json.loads(raw_body))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid payload: {exc}",
        ) from exc

    logger.info(
        "Webhook event=%r product_ids=%s source=%s",
        payload.event,
        payload.product_ids,
        payload.source,
    )

    background_tasks.add_task(_reindex_products, payload.product_ids, settings)

    return WebhookResponse(accepted=True, queued_products=len(payload.product_ids))

"""Order endpoints: place order, track shipment, cancel."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.order import (
    CancelRequest,
    CancelResponse,
    OrderCreate,
    OrderResponse,
    OrderStatus,
    TrackingInfo,
)
from app.dependencies import AppSettings, DBSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["orders"])


# ---------------------------------------------------------------------------
# Place order
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Place a new order",
)
async def create_order(
    body: OrderCreate,
    db: DBSession,
    settings: AppSettings,
) -> OrderResponse:
    """
    Validates product availability, reserves inventory, and creates a new order
    record.  If ``session_id`` is supplied it is linked to the order for
    conversation-to-conversion analytics.
    """
    # TODO: validate products, check stock, calculate totals, persist to DB
    logger.info(
        "New order request: %d item(s) session=%s",
        len(body.items),
        body.session_id,
    )
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Order creation not yet implemented.",
    )


# ---------------------------------------------------------------------------
# Track order
# ---------------------------------------------------------------------------


@router.get(
    "/{order_id}/track",
    response_model=TrackingInfo,
    summary="Get shipment tracking timeline for an order",
)
async def track_order(
    order_id: UUID,
    db: DBSession,
) -> TrackingInfo:
    """
    Returns the full tracking event timeline for the given order, including
    estimated delivery date and current carrier status.
    """
    # TODO: query order from DB, fetch carrier tracking events
    logger.info("Tracking requested for order %s", order_id)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Order '{order_id}' not found.",
    )


# ---------------------------------------------------------------------------
# Cancel order
# ---------------------------------------------------------------------------


@router.post(
    "/{order_id}/cancel",
    response_model=CancelResponse,
    summary="Cancel an order (only if status allows cancellation)",
)
async def cancel_order(
    order_id: UUID,
    body: CancelRequest,
    db: DBSession,
) -> CancelResponse:
    """
    Cancels the order if it is still in a cancellable state (``pending`` or
    ``confirmed``).  Returns refund eligibility based on current order status.
    """
    # TODO: fetch order, validate cancellable states, update DB, trigger refund
    logger.info("Cancel requested for order %s: reason=%r", order_id, body.reason)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Order '{order_id}' not found.",
    )

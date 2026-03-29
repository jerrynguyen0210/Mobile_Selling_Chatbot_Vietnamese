"""Pydantic schemas for order endpoints."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


class OrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


# ---------------------------------------------------------------------------
# Create order
# ---------------------------------------------------------------------------


class OrderItem(BaseModel):
    product_id: str
    quantity: int = Field(..., ge=1)


class ShippingAddress(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=200)
    phone: str = Field(..., min_length=9, max_length=15)
    address_line: str = Field(..., min_length=1, max_length=500)
    ward: str | None = None
    district: str = Field(..., min_length=1)
    province: str = Field(..., min_length=1)
    postal_code: str | None = None


class OrderCreate(BaseModel):
    """Request body for placing a new order."""

    session_id: UUID | None = Field(
        default=None,
        description="Chat session that led to this order (for analytics)",
    )
    items: list[OrderItem] = Field(..., min_length=1)
    shipping_address: ShippingAddress
    customer_email: EmailStr | None = None
    note: str | None = Field(default=None, max_length=1000)


# ---------------------------------------------------------------------------
# Order response
# ---------------------------------------------------------------------------


class OrderItemDetail(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    unit_price: Decimal
    total_price: Decimal
    currency: str = "VND"


class OrderResponse(BaseModel):
    """Returned after an order is created or retrieved."""

    order_id: UUID = Field(default_factory=uuid4)
    status: OrderStatus = OrderStatus.PENDING
    items: list[OrderItemDetail]
    shipping_address: ShippingAddress
    subtotal: Decimal
    shipping_fee: Decimal = Decimal("0")
    total: Decimal
    currency: str = "VND"
    customer_email: str | None = None
    note: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class TrackingEvent(BaseModel):
    timestamp: datetime
    status: OrderStatus
    location: str | None = None
    description: str


class TrackingInfo(BaseModel):
    """Full shipment tracking timeline for an order."""

    order_id: UUID
    current_status: OrderStatus
    tracking_number: str | None = None
    carrier: str | None = None
    estimated_delivery: datetime | None = None
    events: list[TrackingEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class CancelRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class CancelResponse(BaseModel):
    order_id: UUID
    status: OrderStatus
    cancelled_at: datetime = Field(default_factory=datetime.utcnow)
    refund_eligible: bool = False

"""Top-level APIRouter that aggregates all versioned sub-routers."""

from fastapi import APIRouter

from app.api.v1 import chat, health, orders, products, webhooks

api_router = APIRouter()

# ---------------------------------------------------------------------------
# v1 REST endpoints
# ---------------------------------------------------------------------------

api_router.include_router(health.router, prefix="/health")
api_router.include_router(chat.router, prefix="/v1/chat")
api_router.include_router(products.router, prefix="/v1/products")
api_router.include_router(orders.router, prefix="/v1/orders")
api_router.include_router(webhooks.router, prefix="/v1/webhooks")

"""FastAPI application factory with lifespan, CORS, and middleware registration."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import get_settings
from app.exceptions import register_exception_handlers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info(
        "Starting Mobile Selling Chatbot API (model=%s, rag=%s)",
        settings.claude_model,
        settings.enable_rag,
    )

    # --- warm-up hooks (add service init here as the project grows) ----------
    # e.g. await warm_up_embedding_model()
    # e.g. await verify_qdrant_collection()

    yield  # application is running

    # --- teardown -------------------------------------------------------------
    logger.info("Shutting down Mobile Selling Chatbot API")

    # Close shared Redis client if it was created
    from app.dependencies import _redis_client  # noqa: PLC0415

    if _redis_client is not None:
        await _redis_client.aclose()

    # Dispose SQLAlchemy engine if it was created
    from app.dependencies import _async_engine  # noqa: PLC0415

    if _async_engine is not None:
        await _async_engine.dispose()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Mobile Selling Chatbot Vietnamese",
        description="AI-powered mobile phone selling chatbot for the Vietnamese market.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --- CORS -----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Response compression -------------------------------------------------
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # --- Exception handlers ---------------------------------------------------
    register_exception_handlers(app)

    # --- REST routers ---------------------------------------------------------
    from app.api.router import api_router  # noqa: PLC0415

    app.include_router(api_router, prefix="/api")

    # --- WebSocket ------------------------------------------------------------
    from app.api.ws import router as ws_router  # noqa: PLC0415

    app.include_router(ws_router, prefix="/ws")

    return app


# ---------------------------------------------------------------------------
# Entry-point (uvicorn back_end.app.main:app)
# ---------------------------------------------------------------------------

app = create_app()

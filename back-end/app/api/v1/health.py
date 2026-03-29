"""Liveness and readiness health checks for all dependencies."""

import logging
import time

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.dependencies import AppSettings, DBSession, RedisClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _check_postgres(db: DBSession) -> dict[str, object]:
    start = time.monotonic()
    try:
        from sqlalchemy import text

        await db.execute(text("SELECT 1"))
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 2)}
    except Exception as exc:
        logger.warning("Postgres health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


async def _check_redis(redis: RedisClient) -> dict[str, object]:
    start = time.monotonic()
    try:
        await redis.ping()
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 2)}
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


async def _check_qdrant(settings: AppSettings) -> dict[str, object]:
    start = time.monotonic()
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        await client.get_collection(settings.qdrant_collection)
        await client.close()
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 2)}
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/live", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    """Returns 200 immediately — signals the process is alive."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def readiness(
    db: DBSession,
    redis: RedisClient,
    settings: AppSettings,
) -> JSONResponse:
    """Checks all downstream dependencies and returns 200 only when all are healthy."""
    postgres = await _check_postgres(db)
    redis_check = await _check_redis(redis)
    qdrant = await _check_qdrant(settings)

    checks = {
        "postgres": postgres,
        "redis": redis_check,
        "qdrant": qdrant,
    }

    all_ok = all(v["status"] == "ok" for v in checks.values())
    http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=http_status,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )

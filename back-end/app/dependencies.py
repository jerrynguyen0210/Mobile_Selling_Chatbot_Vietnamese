"""FastAPI dependency injection: DB sessions, Redis client, service singletons."""

from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_async_engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine(settings: Settings):  # type: ignore[no-untyped-def]
    """Lazily create the SQLAlchemy async engine."""
    global _async_engine, _async_session_factory
    if _async_engine is None:
        # Convert sync postgresql:// URL to async postgresql+asyncpg://
        db_url = settings.database_url.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
        _async_engine = create_async_engine(
            db_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            echo=settings.log_level == "DEBUG",
            future=True,
        )
        _async_session_factory = async_sessionmaker(
            _async_engine, expire_on_commit=False
        )
    return _async_engine, _async_session_factory


async def get_db_session(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a SQLAlchemy async session, rolling back on error."""
    _, session_factory = _get_engine(settings)
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

_redis_client: aioredis.Redis | None = None  # type: ignore[type-arg]


async def get_redis(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[aioredis.Redis, None]:  # type: ignore[type-arg]
    """Yield a Redis async client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    yield _redis_client


# ---------------------------------------------------------------------------
# Convenience type aliases for route signatures
# ---------------------------------------------------------------------------

DBSession = Annotated[AsyncSession, Depends(get_db_session)]
RedisClient = Annotated[aioredis.Redis, Depends(get_redis)]  # type: ignore[type-arg]
AppSettings = Annotated[Settings, Depends(get_settings)]

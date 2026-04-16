"""Application configuration via Pydantic Settings.

All settings are read from environment variables (or a .env file).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Single source of truth: repo-root .env
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # LLM — Anthropic Claude
    # -------------------------------------------------------------------------
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    claude_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model ID used for chat completions",
    )
    claude_max_tokens: int = Field(default=1024, ge=1, le=8192)

    # -------------------------------------------------------------------------
    # Cache — Redis
    # -------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://:redis_secret@localhost:6379/0",
        description="Redis connection URL",
    )
    session_ttl: int = Field(default=86400, ge=60, description="Session TTL in seconds")

    # -------------------------------------------------------------------------
    # Vector Store — Qdrant
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="mobile_products")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base"
    )
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # Backend API
    # -------------------------------------------------------------------------
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000, ge=1, le=65535)
    secret_key: str = Field(..., description="Secret key for signing JWT / session tokens")
    cors_origins: list[str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # -------------------------------------------------------------------------
    # Feature flags
    # -------------------------------------------------------------------------
    enable_rag: bool = Field(default=True, description="Enable RAG pipeline")
    enable_conversation_history: bool = Field(
        default=True, description="Enable conversation history endpoint"
    )
    enable_cache: bool = Field(default=True, description="Use Redis response cache")

    # -------------------------------------------------------------------------
    # Data ingestion
    # -------------------------------------------------------------------------
    product_data_path: str = Field(default="data/products.json")
    ingest_batch_size: int = Field(default=50, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        """Accept either a list or a comma-separated string from env vars."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()

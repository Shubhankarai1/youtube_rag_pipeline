"""Application configuration and environment loading."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AppConfig(BaseModel):
    """Typed application settings used across the MVP."""

    model_config = ConfigDict(frozen=True)

    openai_api_key: str = Field(min_length=1)
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    database_url: str = Field(min_length=1)
    app_env: str = "development"
    log_level: str = "INFO"
    top_k_results: int = 5
    retrieval_candidates: int = 12
    similarity_threshold: float = 0.3
    max_chunks_per_source: int = 2
    chunk_max_tokens: int = 500
    max_question_chars: int = 500
    min_question_interval_seconds: float = 2.0
    max_context_chars: int = 6000


# ✅ Keep this simple and independent of cached config
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DATABASE_URL"]


def get_missing_required_settings() -> list[str]:
    """Return any required environment variables that are currently missing."""
    missing = []
    for key in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if value is None or value.strip() == "":
            missing.append(key)
    return missing


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and validate environment configuration."""

    # Load .env first
    load_dotenv()

    # Validate required variables
    missing = get_missing_required_settings()
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    # Build config
    return AppConfig(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        database_url=os.environ["DATABASE_URL"],
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "5")),
        retrieval_candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "12")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        max_chunks_per_source=int(os.getenv("MAX_CHUNKS_PER_SOURCE", "2")),
        chunk_max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "500")),
        max_question_chars=int(os.getenv("MAX_QUESTION_CHARS", "500")),
        min_question_interval_seconds=float(os.getenv("MIN_QUESTION_INTERVAL_SECONDS", "2.0")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "6000")),
    )

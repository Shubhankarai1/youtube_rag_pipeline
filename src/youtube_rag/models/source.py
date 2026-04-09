"""Source registry models for persistent knowledge storage."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Supported source types for indexed knowledge."""

    YOUTUBE = "youtube"
    DOCUMENT = "document"


class SourceProcessingStatus(str, Enum):
    """Lifecycle states for indexed sources."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SourceRecord(BaseModel):
    """Persistent source registry entry."""

    source_id: str = Field(min_length=1)
    source_type: SourceType
    external_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    processing_status: SourceProcessingStatus
    source_url: str | None = None
    normalized_url: str | None = None

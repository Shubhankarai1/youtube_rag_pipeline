"""Transcript extraction and normalization models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TranscriptProcessingStatus(str, Enum):
    """High-level transcript extraction outcomes."""

    READY = "ready"
    NOT_AVAILABLE = "not_available"
    TRANSCRIPTS_DISABLED = "transcripts_disabled"
    EMPTY = "empty"
    EXTRACTION_ERROR = "extraction_error"


class TranscriptSegment(BaseModel):
    """Normalized transcript segment used by downstream phases."""

    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    duration: float = Field(ge=0)


class TranscriptMetadata(BaseModel):
    """Summary statistics for extracted transcript content."""

    language: str
    language_code: str
    is_generated: bool
    total_segments: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)


class TranscriptPayload(BaseModel):
    """Persistence-ready transcript payload for chunking and storage."""

    video_id: str
    segments: list[TranscriptSegment]
    metadata: TranscriptMetadata


class TranscriptExtractionResponse(BaseModel):
    """Structured transcript extraction result returned to callers."""

    success: bool
    status: TranscriptProcessingStatus
    message: str
    payload: TranscriptPayload | None = None

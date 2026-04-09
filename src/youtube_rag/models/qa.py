"""Question answering request and response models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from youtube_rag.models.chunk import RetrievedChunk


class QAStatus(str, Enum):
    """High-level answer generation outcomes."""

    ANSWERED = "answered"
    IRRELEVANT = "irrelevant"
    NO_CONTEXT = "no_context"
    ERROR = "error"


class QARequest(BaseModel):
    """Question request against all content or a selected source scope."""

    video_id: str | None = None
    selected_source_ids: list[str] = Field(default_factory=list)
    question: str = Field(min_length=1, max_length=500)


class QAResponse(BaseModel):
    """Structured answer result returned to the UI."""

    success: bool
    status: QAStatus
    message: str
    answer: str | None = None
    sources: list[RetrievedChunk] = Field(default_factory=list)

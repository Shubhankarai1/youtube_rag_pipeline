"""Video intake models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VideoAvailabilityStatus(str, Enum):
    """Availability states produced during intake."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class VideoProcessingStatus(str, Enum):
    """High-level intake outcomes returned to the UI and downstream phases."""

    READY = "ready"
    INVALID_URL = "invalid_url"
    DUPLICATE = "duplicate"
    UNAVAILABLE = "unavailable"


class VideoIntakeRequest(BaseModel):
    """Raw user input for the intake step."""

    youtube_url: str = Field(min_length=1)


class VideoIntakePayload(BaseModel):
    """Normalized video descriptor handed to transcript extraction."""

    video_id: str
    source_url: str
    normalized_url: str


class VideoIntakeResponse(BaseModel):
    """Structured response for Phase 1 intake."""

    accepted: bool
    status: VideoProcessingStatus
    message: str
    availability_status: VideoAvailabilityStatus
    is_duplicate: bool = False
    payload: VideoIntakePayload | None = None

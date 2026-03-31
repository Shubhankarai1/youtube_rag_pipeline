"""Video URL validation, metadata extraction, and deduplication checks."""

from __future__ import annotations

import logging
from typing import Protocol

from youtube_rag.models.video import (
    VideoAvailabilityStatus,
    VideoIntakeRequest,
    VideoIntakeResponse,
    VideoProcessingStatus,
)
from youtube_rag.utils.youtube import build_intake_payload

logger = logging.getLogger(__name__)


class DuplicateVideoRepository(Protocol):
    """Storage contract for duplicate detection during intake."""

    def exists(self, video_id: str) -> bool:
        """Return True if the video has already been ingested."""

    def mark_processed(self, video_id: str) -> None:
        """Persist that the video is now in the pipeline."""


class VideoAvailabilityChecker(Protocol):
    """Availability contract kept separate from later transcript extraction."""

    def check(self, video_id: str) -> VideoAvailabilityStatus:
        """Return the current availability status for a video."""


class InMemoryVideoRegistry:
    """Simple duplicate registry used for tests and local MVP work."""

    def __init__(self, existing_video_ids: set[str] | None = None) -> None:
        self._video_ids = set(existing_video_ids or set())

    def exists(self, video_id: str) -> bool:
        return video_id in self._video_ids

    def mark_processed(self, video_id: str) -> None:
        self._video_ids.add(video_id)


class StaticAvailabilityChecker:
    """Deterministic availability checker for Phase 1 and tests."""

    def __init__(
        self,
        unavailable_video_ids: set[str] | None = None,
        default_status: VideoAvailabilityStatus = VideoAvailabilityStatus.AVAILABLE,
    ) -> None:
        self._unavailable_video_ids = set(unavailable_video_ids or set())
        self._default_status = default_status

    def check(self, video_id: str) -> VideoAvailabilityStatus:
        if video_id in self._unavailable_video_ids:
            return VideoAvailabilityStatus.UNAVAILABLE
        return self._default_status


class VideoIngestionService:
    """Phase 1 video intake orchestration."""

    def __init__(
        self,
        duplicate_repository: DuplicateVideoRepository,
        availability_checker: VideoAvailabilityChecker,
    ) -> None:
        self._duplicate_repository = duplicate_repository
        self._availability_checker = availability_checker

    def intake(self, request: VideoIntakeRequest) -> VideoIntakeResponse:
        """Validate, normalize, deduplicate, and gate a YouTube video submission."""

        try:
            payload = build_intake_payload(request.youtube_url)
        except ValueError:
            logger.info("Rejected invalid YouTube URL", extra={"youtube_url": request.youtube_url})
            return VideoIntakeResponse(
                accepted=False,
                status=VideoProcessingStatus.INVALID_URL,
                message="Enter a valid YouTube video URL before processing.",
                availability_status=VideoAvailabilityStatus.UNKNOWN,
            )

        if self._duplicate_repository.exists(payload.video_id):
            logger.info("Rejected duplicate video submission", extra={"video_id": payload.video_id})
            return VideoIntakeResponse(
                accepted=False,
                status=VideoProcessingStatus.DUPLICATE,
                message="This video has already been submitted for processing.",
                availability_status=VideoAvailabilityStatus.AVAILABLE,
                is_duplicate=True,
                payload=payload,
            )

        availability_status = self._availability_checker.check(payload.video_id)
        if availability_status == VideoAvailabilityStatus.UNAVAILABLE:
            logger.warning("Rejected unavailable video", extra={"video_id": payload.video_id})
            return VideoIntakeResponse(
                accepted=False,
                status=VideoProcessingStatus.UNAVAILABLE,
                message="This video is currently unavailable for processing.",
                availability_status=availability_status,
                payload=payload,
            )

        self._duplicate_repository.mark_processed(payload.video_id)
        logger.info("Accepted video for downstream transcript processing", extra={"video_id": payload.video_id})
        return VideoIntakeResponse(
            accepted=True,
            status=VideoProcessingStatus.READY,
            message="Video intake complete. Ready for transcript extraction.",
            availability_status=availability_status,
            payload=payload,
        )

"""Video URL validation, metadata extraction, and idempotent source intake."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from youtube_rag.models.video import (
    VideoAvailabilityStatus,
    VideoIntakeRequest,
    VideoIntakeResponse,
    VideoProcessingStatus,
)
from youtube_rag.models.source import SourceProcessingStatus
from youtube_rag.utils.youtube import build_intake_payload

logger = logging.getLogger(__name__)


class VideoSourceRepository(Protocol):
    """Storage contract for durable, source-aware intake state."""

    def get_status(self, video_id: str) -> SourceProcessingStatus | None:
        """Return the current persisted processing state for a video."""

    def start_processing(self, *, video_id: str, source_url: str, normalized_url: str) -> None:
        """Persist that the video has entered the processing pipeline."""

    def mark_ready(self, video_id: str) -> None:
        """Persist that the video has been indexed successfully."""

    def mark_failed(self, video_id: str) -> None:
        """Persist that processing failed and the source is retryable."""

    def list_ready_sources(self) -> list[object]:
        """Return ready sources available for retrieval."""


class VideoAvailabilityChecker(Protocol):
    """Availability contract kept separate from later transcript extraction."""

    def check(self, video_id: str) -> VideoAvailabilityStatus:
        """Return the current availability status for a video."""


class InMemoryVideoRegistry:
    """In-memory source registry used for tests and local fallback flows."""

    def __init__(self, existing_video_ids: set[str] | None = None) -> None:
        self._statuses = {
            video_id: SourceProcessingStatus.READY for video_id in (existing_video_ids or set())
        }

    def get_status(self, video_id: str) -> SourceProcessingStatus | None:
        return self._statuses.get(video_id)

    def start_processing(self, *, video_id: str, source_url: str, normalized_url: str) -> None:
        self._statuses[video_id] = SourceProcessingStatus.PROCESSING

    def mark_ready(self, video_id: str) -> None:
        self._statuses[video_id] = SourceProcessingStatus.READY

    def mark_failed(self, video_id: str) -> None:
        self._statuses[video_id] = SourceProcessingStatus.FAILED

    def exists(self, video_id: str) -> bool:
        return self.get_status(video_id) == SourceProcessingStatus.READY

    def list_ready_sources(self) -> list[object]:
        from youtube_rag.models.source import SourceRecord, SourceType

        return [
            SourceRecord(
                source_id=uuid5(NAMESPACE_URL, f"youtube:{video_id}"),
                source_type=SourceType.YOUTUBE,
                external_id=video_id,
                title=f"YouTube Video {video_id}",
                processing_status=SourceProcessingStatus.READY,
                source_url=f"https://www.youtube.com/watch?v={video_id}",
                normalized_url=f"https://www.youtube.com/watch?v={video_id}",
            )
            for video_id, status in sorted(self._statuses.items())
            if status == SourceProcessingStatus.READY
        ]


class FileBackedVideoRegistry:
    """Persist per-video processing state for local MVP work."""

    def __init__(self, storage_path: str | Path) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._statuses = self._load_statuses()

    def get_status(self, video_id: str) -> SourceProcessingStatus | None:
        return self._statuses.get(video_id)

    def start_processing(self, *, video_id: str, source_url: str, normalized_url: str) -> None:
        self._statuses[video_id] = SourceProcessingStatus.PROCESSING
        self._persist()

    def mark_ready(self, video_id: str) -> None:
        self._statuses[video_id] = SourceProcessingStatus.READY
        self._persist()

    def mark_failed(self, video_id: str) -> None:
        self._statuses[video_id] = SourceProcessingStatus.FAILED
        self._persist()

    def exists(self, video_id: str) -> bool:
        return self.get_status(video_id) == SourceProcessingStatus.READY

    def list_ready_sources(self) -> list[object]:
        from youtube_rag.models.source import SourceRecord, SourceType

        return [
            SourceRecord(
                source_id=uuid5(NAMESPACE_URL, f"youtube:{video_id}"),
                source_type=SourceType.YOUTUBE,
                external_id=video_id,
                title=f"YouTube Video {video_id}",
                processing_status=SourceProcessingStatus.READY,
                source_url=f"https://www.youtube.com/watch?v={video_id}",
                normalized_url=f"https://www.youtube.com/watch?v={video_id}",
            )
            for video_id, status in sorted(self._statuses.items())
            if status == SourceProcessingStatus.READY
        ]

    def _load_statuses(self) -> dict[str, SourceProcessingStatus]:
        if not self._storage_path.exists():
            return {}

        try:
            import json

            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning(
                "Could not load duplicate video registry, starting fresh",
                extra={"storage_path": str(self._storage_path)},
            )
            return {}

        if isinstance(payload, list):
            return {
                str(video_id): SourceProcessingStatus.READY
                for video_id in payload
                if str(video_id).strip()
            }

        if not isinstance(payload, dict):
            logger.warning(
                "Unexpected duplicate video registry format, starting fresh",
                extra={"storage_path": str(self._storage_path)},
            )
            return {}

        loaded_statuses: dict[str, SourceProcessingStatus] = {}
        for video_id, raw_status in payload.items():
            try:
                loaded_statuses[str(video_id)] = SourceProcessingStatus(str(raw_status))
            except ValueError:
                continue

        return loaded_statuses

    def _persist(self) -> None:
        import json

        self._storage_path.write_text(
            json.dumps({video_id: status.value for video_id, status in sorted(self._statuses.items())}, indent=2),
            encoding="utf-8",
        )


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
    """Phase 8 source-aware video intake orchestration."""

    def __init__(
        self,
        source_repository: VideoSourceRepository,
        availability_checker: VideoAvailabilityChecker,
    ) -> None:
        self._source_repository = source_repository
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

        current_status = self._source_repository.get_status(payload.video_id)
        if current_status == SourceProcessingStatus.READY:
            logger.info("Rejected duplicate video submission", extra={"video_id": payload.video_id})
            return VideoIntakeResponse(
                accepted=False,
                status=VideoProcessingStatus.DUPLICATE,
                message="This video has already been indexed and is ready for questions.",
                availability_status=VideoAvailabilityStatus.AVAILABLE,
                is_duplicate=True,
                payload=payload,
            )
        if current_status == SourceProcessingStatus.PROCESSING:
            logger.info("Rejected in-flight video submission", extra={"video_id": payload.video_id})
            return VideoIntakeResponse(
                accepted=False,
                status=VideoProcessingStatus.DUPLICATE,
                message="This video is already being processed. Please wait for indexing to finish.",
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

        self._source_repository.start_processing(
            video_id=payload.video_id,
            source_url=payload.source_url,
            normalized_url=payload.normalized_url,
        )
        logger.info(
            "Accepted video for downstream transcript processing",
            extra={"video_id": payload.video_id, "retry": current_status == SourceProcessingStatus.FAILED},
        )
        return VideoIntakeResponse(
            accepted=True,
            status=VideoProcessingStatus.READY,
            message=(
                "Retrying previously failed video processing."
                if current_status == SourceProcessingStatus.FAILED
                else "Video intake complete. Ready for transcript extraction."
            ),
            availability_status=availability_status,
            payload=payload,
        )

    def mark_ready(self, video_id: str) -> None:
        self._source_repository.mark_ready(video_id)

    def mark_failed(self, video_id: str) -> None:
        self._source_repository.mark_failed(video_id)

    def list_ready_sources(self) -> list[object]:
        return self._source_repository.list_ready_sources()

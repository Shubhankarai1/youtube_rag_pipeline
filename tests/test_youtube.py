"""Tests for YouTube URL parsing and intake validation."""

from youtube_rag.models.video import (
    VideoAvailabilityStatus,
    VideoIntakeRequest,
    VideoProcessingStatus,
)
from youtube_rag.services.video_ingestion import (
    InMemoryVideoRegistry,
    StaticAvailabilityChecker,
    VideoIngestionService,
)
from youtube_rag.utils.youtube import build_intake_payload, extract_video_id, is_valid_youtube_url


def test_extract_video_id_from_supported_urls() -> None:
    video_id = "dQw4w9WgXcQ"
    valid_urls = [
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://youtube.com/watch?v={video_id}&t=42s",
        f"https://m.youtube.com/watch?v={video_id}",
        f"https://youtu.be/{video_id}",
        f"https://www.youtube.com/embed/{video_id}",
        f"https://www.youtube.com/shorts/{video_id}",
        f"https://www.youtube.com/live/{video_id}?feature=share",
    ]

    for url in valid_urls:
        assert extract_video_id(url) == video_id
        assert is_valid_youtube_url(url) is True


def test_invalid_urls_are_rejected() -> None:
    invalid_urls = [
        "",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/",
        "not-a-url",
    ]

    for url in invalid_urls:
        assert extract_video_id(url) is None
        assert is_valid_youtube_url(url) is False


def test_build_intake_payload_normalizes_watch_url() -> None:
    payload = build_intake_payload("https://youtu.be/dQw4w9WgXcQ?t=10")

    assert payload.video_id == "dQw4w9WgXcQ"
    assert payload.normalized_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_duplicate_video_detection_logic() -> None:
    duplicate_repository = InMemoryVideoRegistry({"dQw4w9WgXcQ"})
    service = VideoIngestionService(
        duplicate_repository=duplicate_repository,
        availability_checker=StaticAvailabilityChecker(),
    )

    response = service.intake(
        VideoIntakeRequest(youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    )

    assert response.accepted is False
    assert response.status == VideoProcessingStatus.DUPLICATE
    assert response.is_duplicate is True


def test_unavailable_video_is_rejected() -> None:
    unavailable_video_id = "dQw4w9WgXcQ"
    service = VideoIngestionService(
        duplicate_repository=InMemoryVideoRegistry(),
        availability_checker=StaticAvailabilityChecker({unavailable_video_id}),
    )

    response = service.intake(
        VideoIntakeRequest(youtube_url=f"https://youtu.be/{unavailable_video_id}")
    )

    assert response.accepted is False
    assert response.status == VideoProcessingStatus.UNAVAILABLE
    assert response.availability_status == VideoAvailabilityStatus.UNAVAILABLE


def test_valid_video_is_accepted_and_marked_processed() -> None:
    repository = InMemoryVideoRegistry()
    service = VideoIngestionService(
        duplicate_repository=repository,
        availability_checker=StaticAvailabilityChecker(),
    )

    response = service.intake(
        VideoIntakeRequest(youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    )

    assert response.accepted is True
    assert response.status == VideoProcessingStatus.READY
    assert response.payload is not None
    assert repository.exists("dQw4w9WgXcQ") is True

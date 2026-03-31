"""Tests for transcript extraction and normalization behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
)

from youtube_rag.models.transcript import TranscriptProcessingStatus
from youtube_rag.services.transcript_service import TranscriptService


@dataclass
class FakeSnippet:
    text: str
    start: float
    duration: float


@dataclass
class FakeFetchedTranscript:
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    snippets: list[FakeSnippet]


class FakeTranscriptClient:
    def __init__(self, transcript: FakeFetchedTranscript | None = None, error: Exception | None = None) -> None:
        self._transcript = transcript
        self._error = error

    def fetch(self, video_id: str, languages=("en",), preserve_formatting: bool = False) -> FakeFetchedTranscript:
        if self._error is not None:
            raise self._error
        assert self._transcript is not None
        return self._transcript


def test_extract_normalizes_segments_and_metadata() -> None:
    service = TranscriptService(
        transcript_client=FakeTranscriptClient(
            transcript=FakeFetchedTranscript(
                video_id="dQw4w9WgXcQ",
                language="English",
                language_code="en",
                is_generated=False,
                snippets=[
                    FakeSnippet(text="  second line  ", start=4.5, duration=1.25),
                    FakeSnippet(text=" ", start=2.0, duration=1.0),
                    FakeSnippet(text="first line", start=0.0, duration=4.0),
                ],
            )
        )
    )

    response = service.extract("dQw4w9WgXcQ")

    assert response.success is True
    assert response.status == TranscriptProcessingStatus.READY
    assert response.payload is not None
    assert [segment.text for segment in response.payload.segments] == ["first line", "second line"]
    assert response.payload.metadata.total_segments == 2
    assert response.payload.metadata.total_duration_seconds == pytest.approx(5.75)


def test_extract_returns_not_available_when_no_transcript_found() -> None:
    service = TranscriptService(transcript_client=FakeTranscriptClient(error=NoTranscriptFound("video", [], None)))

    response = service.extract("dQw4w9WgXcQ")

    assert response.success is False
    assert response.status == TranscriptProcessingStatus.NOT_AVAILABLE
    assert response.payload is None


def test_extract_returns_transcripts_disabled_status() -> None:
    service = TranscriptService(transcript_client=FakeTranscriptClient(error=TranscriptsDisabled("video")))

    response = service.extract("dQw4w9WgXcQ")

    assert response.success is False
    assert response.status == TranscriptProcessingStatus.TRANSCRIPTS_DISABLED
    assert response.payload is None


def test_extract_returns_empty_when_all_segments_are_blank() -> None:
    service = TranscriptService(
        transcript_client=FakeTranscriptClient(
            transcript=FakeFetchedTranscript(
                video_id="dQw4w9WgXcQ",
                language="English",
                language_code="en",
                is_generated=True,
                snippets=[
                    FakeSnippet(text=" ", start=0.0, duration=1.0),
                    FakeSnippet(text="\n", start=1.0, duration=1.0),
                ],
            )
        )
    )

    response = service.extract("dQw4w9WgXcQ")

    assert response.success is False
    assert response.status == TranscriptProcessingStatus.EMPTY
    assert response.payload is None


def test_extract_returns_error_when_fetch_fails() -> None:
    service = TranscriptService(
        transcript_client=FakeTranscriptClient(error=CouldNotRetrieveTranscript("video"))
    )

    response = service.extract("dQw4w9WgXcQ")

    assert response.success is False
    assert response.status == TranscriptProcessingStatus.EXTRACTION_ERROR
    assert response.payload is None

"""Tests for transcript chunking behavior."""

from __future__ import annotations

import pytest

from youtube_rag.models.transcript import TranscriptMetadata, TranscriptPayload, TranscriptSegment
from youtube_rag.services.chunking_service import ChunkingService


class FakeTokenCounter:
    def __init__(self, token_map: dict[str, int]) -> None:
        self._token_map = token_map

    def count_tokens(self, text: str) -> int:
        return self._token_map[text]


def test_chunk_text_splits_text_when_token_limit_is_exceeded() -> None:
    service = ChunkingService(
        token_counter=FakeTokenCounter(
            {
                "alpha": 1,
                "alpha beta": 2,
                "alpha beta gamma": 3,
                "delta": 1,
            }
        ),
        max_chunk_tokens=2,
    )

    chunks = service.chunk_text("alpha beta gamma delta")

    assert chunks == ["alpha beta gamma", "delta"]


def test_chunk_transcript_creates_word_based_chunks() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="alpha beta gamma", start=0.0, duration=3.0),
            TranscriptSegment(text="delta epsilon", start=3.0, duration=2.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=2,
            total_duration_seconds=5.0,
        ),
    )
    service = ChunkingService(
        token_counter=FakeTokenCounter(
            {
                "alpha": 1,
                "alpha beta": 2,
                "alpha beta gamma": 3,
                "alpha beta gamma delta": 4,
                "epsilon": 1,
            }
        ),
        max_chunk_tokens=3,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "vid123_0001"
    assert chunks[0].text == "alpha beta gamma delta"
    assert chunks[0].token_count == 4
    assert chunks[0].start_time == pytest.approx(0.0)
    assert chunks[0].end_time == pytest.approx(4.0)
    assert [sentence.text for sentence in chunks[0].sentences] == ["alpha beta gamma delta"]
    assert chunks[1].chunk_id == "vid123_0002"
    assert chunks[1].text == "epsilon"
    assert chunks[1].start_time == pytest.approx(4.0)
    assert chunks[1].end_time == pytest.approx(5.0)


def test_chunk_transcript_maps_word_timestamps_within_segment() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="alpha beta", start=10.0, duration=4.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=1,
            total_duration_seconds=14.0,
        ),
    )
    service = ChunkingService(
        token_counter=FakeTokenCounter({"alpha": 1, "alpha beta": 2}),
        max_chunk_tokens=10,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].start_time == pytest.approx(10.0)
    assert chunks[0].end_time == pytest.approx(14.0)
    assert chunks[0].sentences[0].start_time == pytest.approx(10.0)
    assert chunks[0].sentences[0].end_time == pytest.approx(14.0)


def test_chunk_transcript_skips_blank_segments() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text=" ", start=0.0, duration=2.0),
            TranscriptSegment(text="only words", start=2.0, duration=3.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=2,
            total_duration_seconds=5.0,
        ),
    )
    service = ChunkingService(
        token_counter=FakeTokenCounter({"only": 1, "only words": 2}),
        max_chunk_tokens=10,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].text == "only words"
    assert chunks[0].start_time == pytest.approx(2.0)
    assert chunks[0].end_time == pytest.approx(5.0)

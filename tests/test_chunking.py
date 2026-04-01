"""Tests for transcript chunking behavior."""

from __future__ import annotations

import pytest

from youtube_rag.models.transcript import TranscriptMetadata, TranscriptPayload, TranscriptSegment
from youtube_rag.services.chunking_service import ChunkingError, ChunkingService


class FakeTokenCounter:
    def __init__(self, token_map: dict[str, int]) -> None:
        self._token_map = token_map

    def count_tokens(self, text: str) -> int:
        return self._token_map[text]


def test_chunk_transcript_creates_sentence_preserving_chunks() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="First sentence. Second sentence.", start=0.0, duration=6.0),
            TranscriptSegment(text="Third sentence.", start=6.0, duration=2.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=2,
            total_duration_seconds=8.0,
        ),
    )
    service = ChunkingService(
        sentence_splitter=lambda text: [part.strip() for part in text.split(". ") if part.strip()],
        token_counter=FakeTokenCounter(
            {
                "First sentence": 200,
                "Second sentence.": 180,
                "Third sentence.": 150,
            }
        ),
        max_chunk_tokens=512,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "vid123_0001"
    assert chunks[0].text == "First sentence Second sentence."
    assert chunks[0].token_count == 380
    assert chunks[0].start_time == pytest.approx(0.0)
    assert chunks[0].end_time == pytest.approx(6.0)
    assert [sentence.text for sentence in chunks[0].sentences] == ["First sentence", "Second sentence."]
    assert chunks[1].chunk_id == "vid123_0002"
    assert chunks[1].text == "Third sentence."
    assert chunks[1].start_time == pytest.approx(6.0)
    assert chunks[1].end_time == pytest.approx(8.0)


def test_chunk_transcript_maps_sentence_timestamps_within_segment() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="Alpha. Beta.", start=10.0, duration=4.0),
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
        sentence_splitter=lambda text: ["Alpha.", "Beta."],
        token_counter=FakeTokenCounter({"Alpha.": 10, "Beta.": 12}),
        max_chunk_tokens=512,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].start_time == pytest.approx(10.0)
    assert chunks[0].end_time == pytest.approx(14.0)
    assert chunks[0].sentences[0].start_time == pytest.approx(10.0)
    assert chunks[0].sentences[0].end_time < chunks[0].sentences[1].start_time
    assert chunks[0].sentences[1].end_time == pytest.approx(14.0)


def test_chunk_transcript_raises_for_sentence_above_bert_limit() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="Very long sentence.", start=0.0, duration=5.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=1,
            total_duration_seconds=5.0,
        ),
    )
    service = ChunkingService(
        sentence_splitter=lambda text: ["Very long sentence."],
        token_counter=FakeTokenCounter({"Very long sentence.": 513}),
        max_chunk_tokens=512,
    )

    with pytest.raises(ChunkingError):
        service.chunk_transcript(transcript)


def test_chunk_transcript_skips_blank_segments() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text=" ", start=0.0, duration=2.0),
            TranscriptSegment(text="Only sentence.", start=2.0, duration=3.0),
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
        sentence_splitter=lambda text: ["Only sentence."],
        token_counter=FakeTokenCounter({"Only sentence.": 30}),
        max_chunk_tokens=512,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].text == "Only sentence."
    assert chunks[0].start_time == pytest.approx(2.0)
    assert chunks[0].end_time == pytest.approx(5.0)

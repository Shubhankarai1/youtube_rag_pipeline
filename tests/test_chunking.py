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


class FakeSentenceSplitter:
    def __init__(self, sentence_map: dict[str, list[str]]) -> None:
        self._sentence_map = sentence_map

    def split(self, text: str) -> list[str]:
        return self._sentence_map[text]


def test_chunk_text_splits_on_sentence_boundaries_when_token_limit_is_exceeded() -> None:
    service = ChunkingService(
        token_counter=FakeTokenCounter(
            {
                "Alpha.": 2,
                "Beta.": 2,
                "Alpha. Beta.": 4,
            }
        ),
        sentence_splitter=FakeSentenceSplitter({"Alpha. Beta.": ["Alpha.", "Beta."]}),
        max_chunk_tokens=2,
    )

    chunks = service.chunk_text("Alpha. Beta.")

    assert chunks == ["Alpha.", "Beta."]


def test_chunk_transcript_groups_whole_sentences_without_splitting_them() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="Alpha. Beta.", start=0.0, duration=4.0),
            TranscriptSegment(text="Gamma.", start=4.0, duration=2.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=2,
            total_duration_seconds=6.0,
        ),
    )
    service = ChunkingService(
        token_counter=FakeTokenCounter(
            {
                "Alpha.": 2,
                "Beta.": 2,
                "Gamma.": 2,
                "Alpha. Beta.": 4,
                "Alpha. Beta. Gamma.": 6,
            }
        ),
        sentence_splitter=FakeSentenceSplitter(
            {
                "Alpha. Beta.": ["Alpha.", "Beta."],
                "Gamma.": ["Gamma."],
            }
        ),
        max_chunk_tokens=4,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "vid123_0001"
    assert chunks[0].text == "Alpha. Beta."
    assert chunks[0].token_count == 4
    assert chunks[0].start_time == pytest.approx(0.0)
    assert chunks[0].end_time == pytest.approx(4.0)
    assert [sentence.text for sentence in chunks[0].sentences] == ["Alpha.", "Beta."]
    assert chunks[1].chunk_id == "vid123_0002"
    assert chunks[1].text == "Gamma."
    assert chunks[1].start_time == pytest.approx(4.0)
    assert chunks[1].end_time == pytest.approx(6.0)


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
        token_counter=FakeTokenCounter(
            {
                "Alpha.": 2,
                "Beta.": 2,
                "Alpha. Beta.": 4,
            }
        ),
        sentence_splitter=FakeSentenceSplitter({"Alpha. Beta.": ["Alpha.", "Beta."]}),
        max_chunk_tokens=10,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].start_time == pytest.approx(10.0)
    assert chunks[0].end_time == pytest.approx(14.0)
    assert chunks[0].sentences[0].start_time == pytest.approx(10.0)
    assert chunks[0].sentences[0].end_time == pytest.approx(12.0)
    assert chunks[0].sentences[1].start_time == pytest.approx(12.0)
    assert chunks[0].sentences[1].end_time == pytest.approx(14.0)


def test_chunk_transcript_skips_blank_segments() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text=" ", start=0.0, duration=2.0),
            TranscriptSegment(text="Only words.", start=2.0, duration=3.0),
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
                "Only words.": 2,
            }
        ),
        sentence_splitter=FakeSentenceSplitter({"Only words.": ["Only words."]}),
        max_chunk_tokens=10,
    )

    chunks = service.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].text == "Only words."
    assert chunks[0].start_time == pytest.approx(2.0)
    assert chunks[0].end_time == pytest.approx(5.0)


def test_chunk_transcript_rejects_single_sentence_that_exceeds_limit() -> None:
    transcript = TranscriptPayload(
        video_id="vid123",
        segments=[
            TranscriptSegment(text="Very long sentence.", start=0.0, duration=2.0),
        ],
        metadata=TranscriptMetadata(
            language="English",
            language_code="en",
            is_generated=False,
            total_segments=1,
            total_duration_seconds=2.0,
        ),
    )
    service = ChunkingService(
        token_counter=FakeTokenCounter({"Very long sentence.": 12}),
        sentence_splitter=FakeSentenceSplitter({"Very long sentence.": ["Very long sentence."]}),
        max_chunk_tokens=10,
    )

    with pytest.raises(ChunkingError):
        service.chunk_transcript(transcript)

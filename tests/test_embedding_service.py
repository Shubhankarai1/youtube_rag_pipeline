"""Tests for embedding generation and persistence orchestration."""

from __future__ import annotations

from youtube_rag.models.chunk import EmbeddedChunk, TranscriptChunk
from youtube_rag.models.chunk import ChunkSentence
from youtube_rag.models.source import SourceProcessingStatus, SourceRecord, SourceType
from youtube_rag.services.embedding_service import EmbeddingService


class FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.inputs: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return self._vectors


class FakeRepository:
    def __init__(self, has_video: bool = False) -> None:
        self._has_video = has_video
        self.initialized = False
        self.stored: list[EmbeddedChunk] = []
        self.registered_video_ids: list[str] = []

    def initialize_schema(self) -> None:
        self.initialized = True

    def has_video(self, video_id: str) -> bool:
        return self._has_video

    def ensure_youtube_source(self, video_id: str) -> SourceRecord:
        self.registered_video_ids.append(video_id)
        return SourceRecord(
            source_id=f"youtube:{video_id}",
            source_type=SourceType.YOUTUBE,
            external_id=video_id,
            title=f"YouTube Video {video_id}",
            processing_status=SourceProcessingStatus.READY,
        )

    def store_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        self.stored.extend(embedded_chunks)


def _chunk(index: int, text: str) -> TranscriptChunk:
    return TranscriptChunk(
        chunk_id=f"vid123_{index:04d}",
        video_id="vid123",
        text=text,
        start_time=float(index),
        end_time=float(index + 1),
        token_count=10,
        sentences=[
            ChunkSentence(
                text=text,
                start_time=float(index),
                end_time=float(index + 1),
                token_count=10,
            )
        ],
    )


def test_persist_video_chunks_embeds_and_stores_chunks() -> None:
    repository = FakeRepository()
    service = EmbeddingService(
        embedding_client=FakeEmbeddingClient([[0.1, 0.2], [0.3, 0.4]]),
        repository=repository,
    )

    embedded_chunks = service.persist_video_chunks([_chunk(1, "alpha"), _chunk(2, "beta")])

    assert repository.initialized is True
    assert len(repository.stored) == 2
    assert repository.registered_video_ids == ["vid123"]
    assert embedded_chunks[0].source_id == "youtube:vid123"
    assert embedded_chunks[0].embedding == [0.1, 0.2]
    assert embedded_chunks[1].embedding == [0.3, 0.4]


def test_persist_video_chunks_skips_duplicates() -> None:
    repository = FakeRepository(has_video=True)
    embedding_client = FakeEmbeddingClient([[0.1, 0.2]])
    service = EmbeddingService(
        embedding_client=embedding_client,
        repository=repository,
    )

    embedded_chunks = service.persist_video_chunks([_chunk(1, "alpha")])

    assert repository.initialized is True
    assert embedded_chunks == []
    assert repository.stored == []
    assert repository.registered_video_ids == []
    assert embedding_client.inputs == []

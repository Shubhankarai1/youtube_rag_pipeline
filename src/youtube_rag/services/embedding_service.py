"""Embedding generation and persistence orchestration."""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from youtube_rag.models.chunk import EmbeddedChunk, TranscriptChunk


class EmbeddingClient(Protocol):
    """Embedding backend contract."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for the provided texts."""


class ChunkEmbeddingRepository(Protocol):
    """Storage contract for embedded chunks."""

    def initialize_schema(self) -> None:
        """Create or validate the storage schema."""

    def has_video(self, video_id: str) -> bool:
        """Return whether a video's chunks are already stored."""

    def store_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Persist embedded chunks."""


class OpenAIEmbeddingClient:
    """OpenAI-backed embedding generator."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class EmbeddingStorageError(RuntimeError):
    """Raised when chunk embeddings cannot be generated or stored."""


class EmbeddingService:
    """Generate embeddings for chunks and persist them once per video."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        repository: ChunkEmbeddingRepository,
    ) -> None:
        self._embedding_client = embedding_client
        self._repository = repository

    def persist_video_chunks(self, chunks: list[TranscriptChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        video_id = chunks[0].video_id
        try:
            self._repository.initialize_schema()
            if self._repository.has_video(video_id):
                return []

            embeddings = self._embedding_client.embed_texts([chunk.text for chunk in chunks])
            embedded_chunks = [
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    video_id=chunk.video_id,
                    text=chunk.text,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            self._repository.store_embeddings(embedded_chunks)
            return embedded_chunks
        except Exception as exc:  # pragma: no cover - exercised by integration/runtime failures
            raise EmbeddingStorageError(f"Embedding or storage failed: {exc}") from exc


class NullEmbeddingService:
    """Disable Phase 4 persistence when required settings are unavailable."""

    def persist_video_chunks(self, chunks: list[TranscriptChunk]) -> list[EmbeddedChunk]:
        return []

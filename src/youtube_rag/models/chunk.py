"""Chunk and retrieval payload models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkSentence(BaseModel):
    """Sentence-level unit retained inside a retrieval chunk."""

    text: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    token_count: int = Field(ge=1)


class TranscriptChunk(BaseModel):
    """Retrieval-ready transcript chunk built from whole sentences."""

    chunk_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    source_id: str | None = None
    text: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    token_count: int = Field(ge=1)
    sentences: list[ChunkSentence] = Field(min_length=1)


class EmbeddedChunk(BaseModel):
    """Transcript chunk paired with its vector embedding for persistence."""

    chunk_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    token_count: int = Field(ge=1)
    embedding: list[float] = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """Stored chunk returned from vector similarity search."""

    chunk_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    source_id: str | None = None
    source_type: str | None = None
    source_title: str | None = None
    text: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    similarity_score: float

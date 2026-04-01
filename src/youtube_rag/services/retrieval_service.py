"""Vector retrieval and ranking logic."""

from __future__ import annotations

from typing import Protocol

from youtube_rag.models.chunk import RetrievedChunk


class QueryEmbeddingClient(Protocol):
    """Embedding backend for retrieval queries."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for the supplied texts."""


class ChunkRetriever(Protocol):
    """Storage-side retrieval contract."""

    def retrieve_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        similarity_threshold: float,
        video_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most similar stored chunks."""


class RetrievalService:
    """Embed a query and retrieve the best matching stored chunks."""

    def __init__(
        self,
        embedding_client: QueryEmbeddingClient,
        retriever: ChunkRetriever,
        *,
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        self._embedding_client = embedding_client
        self._retriever = retriever
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrieve(self, query: str, *, video_id: str | None = None) -> list[RetrievedChunk]:
        query_embedding = self._embedding_client.embed_texts([query])[0]
        return self._retriever.retrieve_similar_chunks(
            query_embedding,
            top_k=self._top_k,
            similarity_threshold=self._similarity_threshold,
            video_id=video_id,
        )

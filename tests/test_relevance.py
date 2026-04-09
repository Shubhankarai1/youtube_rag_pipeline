"""Tests for retrieval relevance and answerability checks."""

from __future__ import annotations

from youtube_rag.models.chunk import RetrievedChunk
from youtube_rag.services.retrieval_service import RetrievalService


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


class ThresholdAwareRetriever:
    def retrieve_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        similarity_threshold: float,
        video_id: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if similarity_threshold >= 0.95:
            return []
        return [
            RetrievedChunk(
                chunk_id="vid123_0001",
                video_id="vid123",
                text="Relevant transcript evidence.",
                start_time=0.0,
                end_time=5.0,
                similarity_score=0.9,
            )
        ]


def test_retrieval_respects_similarity_threshold_for_relevance_gate() -> None:
    high_threshold_service = RetrievalService(
        embedding_client=FakeEmbeddingClient(),
        retriever=ThresholdAwareRetriever(),
        top_k=5,
        similarity_threshold=0.95,
    )
    low_threshold_service = RetrievalService(
        embedding_client=FakeEmbeddingClient(),
        retriever=ThresholdAwareRetriever(),
        top_k=5,
        similarity_threshold=0.75,
    )

    assert high_threshold_service.retrieve("question", video_id="vid123") == []
    assert len(low_threshold_service.retrieve("question", video_id="vid123")) == 1

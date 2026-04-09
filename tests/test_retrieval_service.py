"""Tests for embedding-backed retrieval orchestration."""

from __future__ import annotations

from youtube_rag.models.chunk import RetrievedChunk
from youtube_rag.services.retrieval_service import RetrievalService


class FakeEmbeddingClient:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.inputs: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return [self._vector]


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def retrieve_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        similarity_threshold: float,
        video_id: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "video_id": video_id,
                "source_ids": source_ids,
            }
        )
        return self.results


def test_retrieve_embeds_query_and_returns_ranked_chunks() -> None:
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                chunk_id="vid123_0001",
                video_id="vid123",
                text="alpha",
                start_time=0.0,
                end_time=5.0,
                similarity_score=0.92,
            )
        ]
    )
    service = RetrievalService(
        embedding_client=FakeEmbeddingClient([0.1, 0.2]),
        retriever=retriever,
        top_k=5,
        similarity_threshold=0.75,
    )

    results = service.retrieve("what is this about?", video_id="vid123")

    assert len(results) == 1
    assert results[0].chunk_id == "vid123_0001"
    assert retriever.calls == [
        {
            "query_embedding": [0.1, 0.2],
            "top_k": 5,
            "similarity_threshold": 0.75,
            "video_id": "vid123",
            "source_ids": None,
        }
    ]


def test_retrieve_supports_selected_source_filters() -> None:
    retriever = FakeRetriever([])
    service = RetrievalService(
        embedding_client=FakeEmbeddingClient([0.3, 0.4]),
        retriever=retriever,
        top_k=3,
        similarity_threshold=0.8,
    )

    service.retrieve("search across selected sources", source_ids=["youtube:vid123", "youtube:vid456"])

    assert retriever.calls == [
        {
            "query_embedding": [0.3, 0.4],
            "top_k": 3,
            "similarity_threshold": 0.8,
            "video_id": None,
            "source_ids": ["youtube:vid123", "youtube:vid456"],
        }
    ]

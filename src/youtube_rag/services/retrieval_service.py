"""Vector retrieval and ranking logic."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Protocol

from youtube_rag.models.chunk import RetrievedChunk


logger = logging.getLogger(__name__)


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
        source_ids: list[str] | None = None,
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
        retrieval_candidates: int | None = None,
        similarity_threshold: float,
        max_chunks_per_source: int = 2,
    ) -> None:
        self._embedding_client = embedding_client
        self._retriever = retriever
        self._top_k = top_k
        self._retrieval_candidates = max(retrieval_candidates or top_k, top_k)
        self._similarity_threshold = similarity_threshold
        self._max_chunks_per_source = max_chunks_per_source

    def retrieve(
        self,
        query: str,
        *,
        video_id: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedding_client.embed_texts([query])[0]
        logger.info(
            "Query embedding generated",
            extra={
                "video_id": video_id,
                "source_ids": source_ids or [],
                "query_length": len(query),
                "embedding_dimensions": len(query_embedding),
            },
        )
        retrieved_chunks = self._retriever.retrieve_similar_chunks(
            query_embedding,
            top_k=self._retrieval_candidates,
            similarity_threshold=self._similarity_threshold,
            video_id=video_id,
            source_ids=source_ids,
        )
        reranked_chunks = self._rerank_chunks(
            query=query,
            chunks=retrieved_chunks,
        )
        logger.info(
            "Retrieved chunks for question",
            extra={
                "video_id": video_id,
                "source_ids": source_ids or [],
                "retrieved_chunk_count": len(reranked_chunks),
                "similarity_scores": [chunk.similarity_score for chunk in reranked_chunks],
            },
        )
        return reranked_chunks

    def _rerank_chunks(self, *, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        query_terms = _tokenize_terms(query)
        scored_chunks = sorted(
            chunks,
            key=lambda chunk: _score_chunk(query_terms, chunk),
            reverse=True,
        )

        selected_chunks: list[RetrievedChunk] = []
        source_counts: Counter[str] = Counter()
        fallback_chunks: list[RetrievedChunk] = []

        for chunk in scored_chunks:
            source_key = chunk.source_id or chunk.video_id
            if source_counts[source_key] >= self._max_chunks_per_source:
                fallback_chunks.append(chunk)
                continue
            selected_chunks.append(chunk)
            source_counts[source_key] += 1
            if len(selected_chunks) == self._top_k:
                return selected_chunks

        for chunk in fallback_chunks:
            selected_chunks.append(chunk)
            if len(selected_chunks) == self._top_k:
                break

        return selected_chunks


def _score_chunk(query_terms: set[str], chunk: RetrievedChunk) -> float:
    chunk_terms = _tokenize_terms(chunk.text)
    if not query_terms or not chunk_terms:
        lexical_overlap = 0.0
    else:
        lexical_overlap = len(query_terms & chunk_terms) / len(query_terms)

    source_bonus = 0.03 if chunk.source_title else 0.0
    return chunk.similarity_score + lexical_overlap + source_bonus


def _tokenize_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(term) > 2}

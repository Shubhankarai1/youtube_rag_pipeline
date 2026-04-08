"""Tests for debug-first question answering."""

from __future__ import annotations

from youtube_rag.models.chunk import RetrievedChunk
from youtube_rag.models.qa import QARequest, QAStatus
from youtube_rag.services.qa_service import QAService


class FakeRetrievalService:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[dict[str, str | None]] = []

    def retrieve(self, query: str, *, video_id: str | None = None) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "video_id": video_id})
        return self._results


class FakeAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[dict[str, object]] = []

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        self.calls.append({"question": question, "chunks": chunks})
        return self._answer


def test_answer_question_returns_grounded_answer_when_context_exists() -> None:
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id="vid123_0001",
            video_id="vid123",
            text="The video explains transformers.",
            start_time=0.0,
            end_time=5.0,
            similarity_score=0.91,
        )
    ]
    retrieval_service = FakeRetrievalService(retrieved_chunks)
    answer_generator = FakeAnswerGenerator("The video explains transformers.")
    service = QAService(retrieval_service=retrieval_service, answer_generator=answer_generator)

    response = service.answer_question(QARequest(video_id="vid123", question="What does the video explain?"))

    assert response.success is True
    assert response.status == QAStatus.ANSWERED
    assert response.answer == "The video explains transformers."
    assert response.sources == retrieved_chunks


def test_answer_question_returns_no_context_when_no_chunks_are_retrieved() -> None:
    service = QAService(
        retrieval_service=FakeRetrievalService([]),
        answer_generator=FakeAnswerGenerator("should not be used"),
    )

    response = service.answer_question(QARequest(video_id="vid123", question="What is the capital of France?"))

    assert response.success is False
    assert response.status == QAStatus.NO_CONTEXT
    assert response.answer is None


def test_answer_question_returns_no_context_when_generator_returns_blank() -> None:
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id="vid123_0001",
            video_id="vid123",
            text="The video mentions attention.",
            start_time=10.0,
            end_time=15.0,
            similarity_score=0.8,
        )
    ]
    service = QAService(
        retrieval_service=FakeRetrievalService(retrieved_chunks),
        answer_generator=FakeAnswerGenerator(" "),
    )

    response = service.answer_question(QARequest(video_id="vid123", question="What is mentioned?"))

    assert response.success is False
    assert response.status == QAStatus.NO_CONTEXT
    assert response.sources == retrieved_chunks

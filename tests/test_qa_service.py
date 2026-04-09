"""Tests for debug-first question answering."""

from __future__ import annotations

from uuid import UUID

from youtube_rag.models.chunk import RetrievedChunk
from youtube_rag.models.qa import QARequest, QAStatus
from youtube_rag.services.qa_service import QAService, _build_context


class FakeRetrievalService:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        *,
        video_id: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "video_id": video_id, "source_ids": source_ids})
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
    assert retrieval_service.calls == [
        {
            "query": "What does the video explain?",
            "video_id": "vid123",
            "source_ids": None,
        }
    ]


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


def test_build_context_truncates_to_max_context_chars() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="vid123_0001",
            video_id="vid123",
            text="A" * 40,
            start_time=0.0,
            end_time=5.0,
            similarity_score=0.95,
        ),
        RetrievedChunk(
            chunk_id="vid123_0002",
            video_id="vid123",
            text="B" * 40,
            start_time=5.0,
            end_time=10.0,
            similarity_score=0.94,
        ),
    ]

    context = _build_context(chunks, max_context_chars=70)

    assert len(context) <= 70
    assert "vid123_0001" in context


def test_answer_question_supports_selected_source_scope() -> None:
    retrieval_service = FakeRetrievalService(
        [
            RetrievedChunk(
                chunk_id="vid123_0001",
                video_id="vid123",
                source_id=UUID("11111111-1111-1111-1111-111111111111"),
                source_type="youtube",
                source_title="YouTube Video vid123",
                text="The video explains agentic retrieval.",
                start_time=0.0,
                end_time=5.0,
                similarity_score=0.93,
            )
        ]
    )
    service = QAService(
        retrieval_service=retrieval_service,
        answer_generator=FakeAnswerGenerator("The video explains agentic retrieval."),
    )

    response = service.answer_question(
        QARequest(
            question="What does this source explain?",
            selected_source_ids=["11111111-1111-1111-1111-111111111111"],
        )
    )

    assert response.success is True
    assert retrieval_service.calls == [
        {
            "query": "What does this source explain?",
            "video_id": None,
            "source_ids": ["11111111-1111-1111-1111-111111111111"],
        }
    ]

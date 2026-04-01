"""Relevance detection and answer generation."""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from youtube_rag.models.chunk import RetrievedChunk
from youtube_rag.models.qa import QARequest, QAResponse, QAStatus
from youtube_rag.services.retrieval_service import RetrievalService


class AnswerGenerator(Protocol):
    """Generate a grounded answer from retrieved context."""

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Return an answer based only on the supplied chunks."""


class OpenAIAnswerGenerator:
    """OpenAI-backed grounded answer generation."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = _build_context(chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided transcript context. "
                        "If the context is insufficient, say so plainly. "
                        "Cite timestamps in parentheses when helpful."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return response.choices[0].message.content or ""


class QAService:
    """Retrieve supporting chunks, reject unsupported questions, and answer grounded ones."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._answer_generator = answer_generator

    def answer_question(self, request: QARequest) -> QAResponse:
        try:
            retrieved_chunks = self._retrieval_service.retrieve(
                request.question,
                video_id=request.video_id,
            )
        except Exception as exc:  # pragma: no cover - integration/runtime path
            return QAResponse(
                success=False,
                status=QAStatus.ERROR,
                message=f"Question answering failed during retrieval: {exc}",
            )

        if not retrieved_chunks:
            return QAResponse(
                success=False,
                status=QAStatus.IRRELEVANT,
                message="This question is not sufficiently supported by the processed video context.",
            )

        try:
            answer = self._answer_generator.generate_answer(request.question, retrieved_chunks).strip()
        except Exception as exc:  # pragma: no cover - integration/runtime path
            return QAResponse(
                success=False,
                status=QAStatus.ERROR,
                message=f"Question answering failed during answer generation: {exc}",
                sources=retrieved_chunks,
            )

        if not answer:
            return QAResponse(
                success=False,
                status=QAStatus.NO_CONTEXT,
                message="The system could not produce an answer from the retrieved transcript context.",
                sources=retrieved_chunks,
            )

        return QAResponse(
            success=True,
            status=QAStatus.ANSWERED,
            message="Answer generated from retrieved transcript context.",
            answer=answer,
            sources=retrieved_chunks,
        )


class NullQAService:
    """Disable Phase 5 answering when configuration is unavailable."""

    def answer_question(self, request: QARequest) -> QAResponse:
        return QAResponse(
            success=False,
            status=QAStatus.ERROR,
            message="Question answering is unavailable because required configuration is missing.",
        )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{chunk.chunk_id} | {chunk.start_time:.2f}-{chunk.end_time:.2f}] {chunk.text}"
        for chunk in chunks
    )

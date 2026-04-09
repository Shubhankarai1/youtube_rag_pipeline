"""UI page-level components and workflow rendering."""

from __future__ import annotations

import time

import streamlit as st

from youtube_rag.models.chunk import TranscriptChunk
from youtube_rag.models.qa import QARequest, QAResponse, QAStatus
from youtube_rag.models.video import VideoIntakeRequest
from youtube_rag.services.chunking_service import ChunkingError, ChunkingService
from youtube_rag.services.embedding_service import EmbeddingService, EmbeddingStorageError, NullEmbeddingService
from youtube_rag.services.qa_service import NullQAService, QAService
from youtube_rag.services.transcript_service import TranscriptService
from youtube_rag.services.video_ingestion import VideoIngestionService


def render_video_intake_page(
    ingestion_service: VideoIngestionService,
    transcript_service: TranscriptService,
    chunking_service: ChunkingService,
    embedding_service: EmbeddingService | NullEmbeddingService,
    qa_service: QAService | NullQAService,
    *,
    max_question_chars: int = 500,
    min_question_interval_seconds: float = 2.0,
    missing_settings: list[str] | None = None,
) -> None:
    """Render the Phase 1-6 Streamlit workflow."""

    st.set_page_config(page_title="YouTube RAG Pipeline", page_icon="film", layout="centered")
    _initialize_ui_state()
    _render_runtime_metrics_sidebar()
    _render_retrieved_context_sidebar(st.session_state.get("latest_qa_response"))
    st.title("YouTube RAG Pipeline")
    st.caption(
        "Phases 1-6: validate a YouTube URL, extract its transcript, build sentence-aware chunks, "
        "store embeddings, answer grounded questions, and expose basic operational safeguards."
    )
    if missing_settings:
        st.warning(
            "Missing required environment variables: "
            + ", ".join(missing_settings)
            + ". Intake is available, but downstream phases will not run yet."
        )

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    process_clicked = st.button("Process Video", type="primary")

    if process_clicked:
        with st.status("Running video intake", expanded=True) as status:
            st.write("Validating the YouTube URL and extracting the `video_id`.")
            response = ingestion_service.intake(VideoIntakeRequest(youtube_url=youtube_url))
            st.write("Checking for duplicate submissions and intake availability.")

            if response.accepted and response.payload:
                st.write("Fetching and normalizing transcript segments.")
                transcript_response = transcript_service.extract(response.payload.video_id)

                if transcript_response.success and transcript_response.payload:
                    st.write("Splitting transcript into sentence-aware BERT-token-limited chunks.")
                    try:
                        chunks = chunking_service.chunk_transcript(transcript_response.payload)
                    except ChunkingError as exc:
                        ingestion_service.mark_failed(response.payload.video_id)
                        _record_metric("processing_errors")
                        status.update(label="Chunking failed", state="error")
                        st.success(f"{response.message} {transcript_response.message}")
                        st.error(str(exc))
                        st.subheader("Video Intake")
                        st.json(_model_to_dict(response))
                        st.subheader("Transcript Metadata")
                        st.json(_model_to_dict(transcript_response.payload.metadata))
                    else:
                        st.write("Generating embeddings and storing chunks in PostgreSQL.")
                        try:
                            embedded_chunks = embedding_service.persist_video_chunks(chunks)
                        except EmbeddingStorageError as exc:
                            ingestion_service.mark_failed(response.payload.video_id)
                            _record_metric("processing_errors")
                            status.update(label="Embedding or storage failed", state="error")
                            st.success(f"{response.message} {transcript_response.message} Chunking complete.")
                            st.error(str(exc))
                            st.subheader("Chunk Preview")
                            st.json([_chunk_to_preview(chunk) for chunk in chunks[:3]])
                        else:
                            ingestion_service.mark_ready(response.payload.video_id)
                            _record_metric("videos_processed")
                            status.update(label="Transcript chunking and storage complete", state="complete")
                            st.success(f"{response.message} {transcript_response.message} Chunking complete.")
                            st.subheader("Video Intake")
                            st.json(_model_to_dict(response))
                            st.subheader("Transcript Metadata")
                            st.json(_model_to_dict(transcript_response.payload.metadata))
                            st.subheader("Transcript Preview")
                            st.json([_model_to_dict(segment) for segment in transcript_response.payload.segments[:5]])
                            st.subheader("Chunk Summary")
                            st.json(
                                {
                                    "total_chunks": len(chunks),
                                    "first_chunk_token_count": chunks[0].token_count if chunks else 0,
                                    "stored_chunks": len(embedded_chunks),
                                    "already_stored": len(chunks) > 0 and len(embedded_chunks) == 0,
                                }
                            )
                            st.subheader("Chunk Preview")
                            st.json([_chunk_to_preview(chunk) for chunk in chunks[:3]])
                            st.session_state["processed_video_id"] = response.payload.video_id
                else:
                    ingestion_service.mark_failed(response.payload.video_id)
                    _record_metric("processing_errors")
                    status.update(label="Transcript extraction failed", state="error")
                    st.success(response.message)
                    st.error(transcript_response.message)
                    st.subheader("Video Intake")
                    st.json(_model_to_dict(response))
                    st.subheader("Transcript Extraction")
                    st.json(_model_to_dict(transcript_response))
            else:
                _record_metric("processing_errors")
                status.update(label="Video intake failed", state="error")
                st.error(response.message)
                if response.payload:
                    st.json(_model_to_dict(response))
                if response.is_duplicate and response.payload:
                    st.session_state["processed_video_id"] = response.payload.video_id
    else:
        st.info("Submit a YouTube URL to start video intake.")

    processed_video_id = st.session_state.get("processed_video_id")
    if processed_video_id:
        st.divider()
        st.subheader("Ask Questions")
        st.caption(f"Current processed video: `{processed_video_id}`")
        st.caption(
            f"Questions are limited to {max_question_chars} characters with a "
            f"{min_question_interval_seconds:.1f}s minimum interval."
        )
        question = st.text_input(
            "Question about the processed video",
            key="qa_question_input",
            placeholder="What is the main topic of this video?",
        )
        ask_clicked = st.button("Ask Question", type="secondary")
        if ask_clicked:
            trimmed_question = question.strip()
            if not trimmed_question:
                st.warning("Enter a question before asking.")
                return
            if len(trimmed_question) > max_question_chars:
                st.warning(f"Question is too long. Keep it under {max_question_chars} characters.")
                return
            if _is_rate_limited(min_question_interval_seconds):
                st.warning("Please wait a moment before sending another question.")
                return

            _record_question_attempt()
            qa_response = qa_service.answer_question(
                QARequest(video_id=processed_video_id, question=trimmed_question)
            )
            st.session_state["latest_qa_response"] = qa_response.model_dump(mode="json")
            _record_qa_result(qa_response)
            _render_qa_response(qa_response)


def _initialize_ui_state() -> None:
    st.session_state.setdefault(
        "ui_metrics",
        {
            "videos_processed": 0,
            "questions_asked": 0,
            "answers_returned": 0,
            "no_context_answers": 0,
            "processing_errors": 0,
            "qa_errors": 0,
            "last_question_at": 0.0,
        },
    )


def _record_metric(metric_name: str) -> None:
    metrics = st.session_state["ui_metrics"]
    metrics[metric_name] = metrics.get(metric_name, 0) + 1


def _record_question_attempt() -> None:
    metrics = st.session_state["ui_metrics"]
    metrics["questions_asked"] = metrics.get("questions_asked", 0) + 1
    metrics["last_question_at"] = time.time()


def _record_qa_result(response: QAResponse) -> None:
    metrics = st.session_state["ui_metrics"]
    if response.status == QAStatus.ANSWERED:
        metrics["answers_returned"] = metrics.get("answers_returned", 0) + 1
    elif response.status in {QAStatus.IRRELEVANT, QAStatus.NO_CONTEXT}:
        metrics["no_context_answers"] = metrics.get("no_context_answers", 0) + 1
    else:
        metrics["qa_errors"] = metrics.get("qa_errors", 0) + 1


def _is_rate_limited(min_interval_seconds: float) -> bool:
    last_question_at = st.session_state["ui_metrics"].get("last_question_at", 0.0)
    return (time.time() - last_question_at) < min_interval_seconds


def _render_runtime_metrics_sidebar() -> None:
    metrics = st.session_state["ui_metrics"]
    st.sidebar.subheader("Runtime Metrics")
    st.sidebar.caption("Session-level operational counters for local MVP monitoring.")
    st.sidebar.metric("Videos Processed", metrics["videos_processed"])
    st.sidebar.metric("Questions Asked", metrics["questions_asked"])
    st.sidebar.metric("Answers Returned", metrics["answers_returned"])
    st.sidebar.metric("Unsupported / Empty", metrics["no_context_answers"])
    st.sidebar.metric("Processing Errors", metrics["processing_errors"])
    st.sidebar.metric("QA Errors", metrics["qa_errors"])
    st.sidebar.divider()


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _chunk_to_preview(chunk: TranscriptChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "video_id": chunk.video_id,
        "start_time": chunk.start_time,
        "end_time": chunk.end_time,
        "token_count": chunk.token_count,
        "sentence_count": len(chunk.sentences),
        "text": chunk.text,
    }


def _render_qa_response(response: QAResponse) -> None:
    if response.status == QAStatus.ANSWERED and response.answer:
        st.success(response.message)
        st.subheader("Answer")
        st.write(response.answer)
    elif response.status in {QAStatus.IRRELEVANT, QAStatus.NO_CONTEXT}:
        st.warning(response.message)
    else:
        st.error(response.message)


def _render_retrieved_context_sidebar(response_data: dict | None) -> None:
    st.sidebar.subheader("Retrieved Context")
    if not response_data or not response_data.get("sources"):
        st.sidebar.caption("Ask a question to inspect the retrieved chunks.")
        return

    for index, chunk in enumerate(response_data["sources"], start=1):
        st.sidebar.markdown(f"**Chunk {index}**")
        similarity_score = chunk.get("similarity_score", 0.0)
        start_time = chunk.get("start_time", 0.0)
        end_time = chunk.get("end_time", 0.0)
        chunk_id = chunk.get("chunk_id", "unknown")
        st.sidebar.caption(
            f"Score: {similarity_score:.3f} | "
            f"Chunk ID: `{chunk_id}` | "
            f"Timestamp: {start_time:.2f}s - {end_time:.2f}s"
        )
        st.sidebar.write(chunk.get("text", ""))
        st.sidebar.divider()

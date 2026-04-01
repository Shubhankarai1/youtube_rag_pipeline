"""UI page-level components and workflow rendering."""

from __future__ import annotations

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
    missing_settings: list[str] | None = None,
) -> None:
    """Render the Phase 1-5 Streamlit workflow."""

    st.set_page_config(page_title="YouTube RAG Pipeline", page_icon="🎥", layout="centered")
    st.title("YouTube RAG Pipeline")
    st.caption(
        "Phases 1-5: validate a YouTube URL, extract its transcript, build sentence-aware chunks, store embeddings, and answer grounded questions."
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
                    st.write("Splitting transcript into sentences and building BERT-sized chunks.")
                    try:
                        chunks = chunking_service.chunk_transcript(transcript_response.payload)
                    except ChunkingError as exc:
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
                            status.update(label="Embedding or storage failed", state="error")
                            st.success(f"{response.message} {transcript_response.message} Chunking complete.")
                            st.error(str(exc))
                            st.subheader("Chunk Preview")
                            st.json([_chunk_to_preview(chunk) for chunk in chunks[:3]])
                        else:
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
                    status.update(label="Transcript extraction failed", state="error")
                    st.success(response.message)
                    st.error(transcript_response.message)
                    st.subheader("Video Intake")
                    st.json(_model_to_dict(response))
                    st.subheader("Transcript Extraction")
                    st.json(_model_to_dict(transcript_response))
            else:
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
        question = st.text_input(
            "Question about the processed video",
            key="qa_question_input",
            placeholder="What is the main topic of this video?",
        )
        ask_clicked = st.button("Ask Question", type="secondary")
        if ask_clicked:
            if not question.strip():
                st.warning("Enter a question before asking.")
                return
            qa_response = qa_service.answer_question(
                QARequest(video_id=processed_video_id, question=question.strip())
            )
            _render_qa_response(qa_response)


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
    elif response.status == QAStatus.IRRELEVANT:
        st.warning(response.message)
    else:
        st.error(response.message)

    if response.sources:
        st.subheader("Retrieved Sources")
        st.json(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "similarity_score": chunk.similarity_score,
                    "text": chunk.text,
                }
                for chunk in response.sources
            ]
        )

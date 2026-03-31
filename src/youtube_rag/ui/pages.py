"""UI page-level components and workflow rendering."""

from __future__ import annotations

import streamlit as st

from youtube_rag.models.video import VideoIntakeRequest
from youtube_rag.services.transcript_service import TranscriptService
from youtube_rag.services.video_ingestion import VideoIngestionService


def render_video_intake_page(
    ingestion_service: VideoIngestionService,
    transcript_service: TranscriptService,
    missing_settings: list[str] | None = None,
) -> None:
    """Render the Phase 1-2 Streamlit workflow."""

    st.set_page_config(page_title="YouTube RAG Pipeline", page_icon="🎥", layout="centered")
    st.title("YouTube RAG Pipeline")
    st.caption("Phases 1-2: validate a YouTube URL, then extract and normalize its transcript.")
    if missing_settings:
        st.warning(
            "Missing required environment variables: "
            + ", ".join(missing_settings)
            + ". Intake is available, but downstream phases will not run yet."
        )

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    process_clicked = st.button("Process Video", type="primary")

    if not process_clicked:
        st.info("Submit a YouTube URL to start video intake.")
        return

    with st.status("Running video intake", expanded=True) as status:
        st.write("Validating the YouTube URL and extracting the `video_id`.")
        response = ingestion_service.intake(VideoIntakeRequest(youtube_url=youtube_url))
        st.write("Checking for duplicate submissions and intake availability.")

        if response.accepted and response.payload:
            st.write("Fetching and normalizing transcript segments.")
            transcript_response = transcript_service.extract(response.payload.video_id)

            if transcript_response.success and transcript_response.payload:
                status.update(label="Transcript extraction complete", state="complete")
                st.success(f"{response.message} {transcript_response.message}")
                st.subheader("Video Intake")
                st.json(_model_to_dict(response))
                st.subheader("Transcript Metadata")
                st.json(_model_to_dict(transcript_response.payload.metadata))
                st.subheader("Transcript Preview")
                st.json([_model_to_dict(segment) for segment in transcript_response.payload.segments[:5]])
                return

            status.update(label="Transcript extraction failed", state="error")
            st.success(response.message)
            st.error(transcript_response.message)
            st.subheader("Video Intake")
            st.json(_model_to_dict(response))
            st.subheader("Transcript Extraction")
            st.json(_model_to_dict(transcript_response))
            return

        status.update(label="Video intake failed", state="error")
        st.error(response.message)
        if response.payload:
            st.json(_model_to_dict(response))


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()

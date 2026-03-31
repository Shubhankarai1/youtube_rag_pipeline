# youtube_rag_pipeline

Build an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to input a YouTube video URL and ask questions based on its content. The system should retrieve relevant transcript chunks and generate accurate answers, or detect and respond when a query is irrelevant.

## Phase 1 status

Phase 1 implements the project shell and video intake flow:

- environment-backed config loading
- logging setup
- Streamlit intake page with a `Process Video` action
- YouTube URL validation and `video_id` extraction
- duplicate detection and deterministic availability checks
- normalized intake payload returned for the transcript layer

## Run locally

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and provide at least `OPENAI_API_KEY` and `DATABASE_URL`.
3. Run `streamlit run streamlit_app.py`.
4. Run `pytest` to verify the Phase 1 tests.

# youtube_rag_pipeline

Build an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to input a YouTube video URL and ask questions based on its content. The system should retrieve relevant transcript chunks and generate accurate answers, or detect and respond when a query is irrelevant.

## Phase 1-5 status

The current repo implements the foundation plus transcript chunking:

- environment-backed config loading
- logging setup
- Streamlit intake page with a `Process Video` action
- YouTube URL validation and `video_id` extraction
- duplicate detection and deterministic availability checks
- normalized intake payload returned for the transcript layer
- transcript extraction and normalization with `youtube-transcript-api`
- sentence-aware chunking with `nltk.sent_tokenize`
- BERT-token-limited chunks using a HuggingFace tokenizer
- chunk metadata with `chunk_id`, `start_time`, `end_time`, and `token_count`
- OpenAI embedding generation for chunks
- PostgreSQL + pgvector persistence for chunk embeddings
- vector retrieval service for top-k semantic search
- retrieval-gated question answering from transcript context only

## Run locally

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and provide at least `OPENAI_API_KEY` and `DATABASE_URL`.
3. Run `streamlit run streamlit_app.py`.
4. Run `python3 -m pytest -q` to verify the tests.

## Phase 3 behavior

- Transcript text is split into sentences with NLTK.
- Each sentence is mapped to timestamps within its source transcript segment.
- Sentences are added to chunks in order until the chunk reaches the configured BERT token limit.
- Sentences are never split across chunks.

## Phase 4 behavior

- Chunks are embedded with the configured OpenAI embedding model.
- The app initializes the `video_chunks` schema in PostgreSQL if needed.
- Processed chunks are stored once per video and skipped on repeat submissions.
- A retrieval service is available to embed user queries and run vector similarity search.

## Phase 5 behavior

- After a video is processed, the app keeps the `video_id` in session state.
- User questions are embedded and matched against stored chunks for that video.
- If retrieval does not produce support above the configured threshold, the question is rejected as unsupported.
- If relevant chunks are found, the chat model answers only from those chunks and the UI shows the retrieved source passages.

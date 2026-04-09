# youtube_rag_pipeline

Build an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to input a YouTube video URL and ask questions based on its content. The system retrieves relevant transcript chunks, generates grounded answers from those chunks, and refuses unsupported questions when retrieval does not meet the configured relevance bar.

## Phase 1-11 status

The current repo now covers the MVP foundation plus advanced continuous-knowledge chat:

- environment-backed config loading
- persistent duplicate detection for processed YouTube videos
- Streamlit intake page with processing status and retrieved-context inspection
- transcript extraction and normalization with `youtube-transcript-api`
- sentence-aware chunking with NLTK
- BERT-token-limited chunks using a HuggingFace tokenizer
- OpenAI embedding generation for chunks
- PostgreSQL + pgvector persistence for chunk embeddings
- top-k semantic retrieval with enforced similarity thresholding
- all-content chat by default with optional selected-source scope
- source-aware retrieval with persistent source registry
- candidate expansion, lightweight reranking, and per-source diversity control
- grounded answer generation from transcript context only
- basic session-level runtime metrics in the sidebar
- question-length and request-interval safeguards for local MVP usage

## Run locally

1. Create a virtual environment.
2. Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.
4. Provide at least:
   - `OPENAI_API_KEY`
   - `DATABASE_URL`
5. Run the app:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

6. Run tests:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## Important environment settings

- `TOP_K_RESULTS`: number of retrieved chunks considered for answers
- `RETRIEVAL_CANDIDATES`: number of initial vector hits fetched before reranking
- `SIMILARITY_THRESHOLD`: minimum similarity required before chunks are eligible
- `MAX_CHUNKS_PER_SOURCE`: diversity cap applied after reranking
- `CHUNK_MAX_TOKENS`: maximum chunk size for sentence-aware chunking
- `MAX_QUESTION_CHARS`: question-length guardrail in the UI and request model
- `MIN_QUESTION_INTERVAL_SECONDS`: local request pacing safeguard
- `MAX_CONTEXT_CHARS`: maximum context size forwarded into answer generation

## Operational behavior

- Duplicate video submissions are persisted in `data/processed_videos.json`.
- The app initializes the `video_chunks` schema automatically when embeddings are first stored.
- The sidebar exposes session-level counters for:
  - videos processed
  - questions asked
  - grounded answers returned
  - unsupported or empty-answer outcomes
  - processing errors
  - QA errors

## MVP limitations

- Duplicate tracking is persistent locally but not yet multi-user aware.
- Rate limiting is session-based and intended as an MVP safeguard, not a production abuse-prevention layer.
- Observability is still log-driven; there is no external metrics backend yet.
- Deployment is still simplest as a single Streamlit app backed by PostgreSQL.

## Deployment notes

Recommended MVP deployment targets:

- Streamlit Community Cloud for a quick demo, if database connectivity is available
- Render or Railway for a more controllable hosted setup

Minimum production-facing setup:

- hosted PostgreSQL with `pgvector`
- environment variables configured in the deployment platform
- outbound access for OpenAI and transcript retrieval
- log capture from the Streamlit runtime

## Verification

The project test suite should pass with:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

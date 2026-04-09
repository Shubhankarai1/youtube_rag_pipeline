# youtube_rag_pipeline

Build an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to input a YouTube video URL and ask questions based on its content. The system retrieves relevant transcript chunks, generates grounded answers from those chunks, and refuses unsupported questions when retrieval does not meet the configured relevance bar.

## Phase 1-11 status

The current repo now covers the MVP foundation plus advanced continuous-knowledge chat:

- environment-backed config loading
- persistent duplicate detection for processed YouTube videos
- Streamlit intake page with processing status, chat history, and retrieved-context inspection
- transcript extraction and normalization with `youtube-transcript-api`
- sentence-aware chunking with NLTK
- BERT-token-limited chunks using a HuggingFace tokenizer
- OpenAI embedding generation for chunks
- PostgreSQL + pgvector persistence for chunk embeddings and source registry records
- top-k semantic retrieval with enforced similarity thresholding
- all-content chat by default with optional selected-source scope
- source-aware retrieval with persistent source registry and UUID source identifiers
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
- The app initializes and upgrades the PostgreSQL schema on startup before the UI queries ready sources.
- The database schema is source-centric:
  - `sources` stores source metadata and processing state
  - `video_chunks` stores transcript chunks, embeddings, and UUID `source_id` references
- The Q&A experience uses Streamlit chat components and keeps session chat history in memory for the active browser session.
- The sidebar exposes session-level counters for:
  - videos processed
  - questions asked
  - grounded answers returned
  - unsupported or empty-answer outcomes
  - processing errors
  - QA errors

## Data model

- `sources`
  - `id` (`UUID`)
  - `source_type` (`TEXT`)
  - `external_id` (`TEXT`)
  - `title` (`TEXT`)
  - `processing_status` (`TEXT`)
  - `source_url` (`TEXT`, nullable)
  - `normalized_url` (`TEXT`, nullable)
  - `created_at` (`TIMESTAMP`)
- `video_chunks`
  - `id` (`UUID`)
  - `video_id` (`TEXT`)
  - `source_id` (`UUID`)
  - `chunk_id` (`TEXT`)
  - `content` (`TEXT`)
  - `embedding` (`VECTOR`)
  - `start_time` (`DOUBLE PRECISION`)
  - `end_time` (`DOUBLE PRECISION`)
  - `created_at` (`TIMESTAMP`)

## MVP limitations

- Duplicate tracking is persistent locally but not yet multi-user aware.
- The fallback local registry remains single-user and is meant for MVP/local development only.
- Rate limiting is session-based and intended as an MVP safeguard, not a production abuse-prevention layer.
- Observability is still log-driven; there is no external metrics backend yet.
- Deployment is still simplest as a single Streamlit app backed by PostgreSQL.
- Document ingestion is still future scope; the current app indexes YouTube sources only.

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

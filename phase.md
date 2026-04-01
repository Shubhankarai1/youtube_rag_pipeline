# YouTube RAG Pipeline Execution Roadmap

## Phase 1: Project Foundation and Video Intake

### Goal
Establish the base application structure and make sure a user can submit a valid YouTube URL that the system can parse and validate reliably.

### Significant Components
- Create the Python project structure for Streamlit UI, backend logic, and shared utilities.
- Set up configuration management for OpenAI keys, database connection strings, and app settings.
- Build the Streamlit input flow with:
  - YouTube URL field
  - `Process Video` trigger
  - status messages for each processing step
- Implement URL validation and `video_id` extraction.
- Add video availability and duplicate-ingestion checks.
- Define the ingestion contract that downstream phases will use.

### Deliverables
- Working Streamlit shell UI.
- Utility functions for URL parsing and validation.
- Structured ingestion request and response models.
- Logging for intake success and failure paths.

### Testing
- Unit tests for:
  - valid YouTube URL parsing
  - invalid URL rejection
  - `video_id` extraction
  - duplicate video detection logic
- Manual tests in Streamlit for:
  - successful processing start
  - invalid links
  - unavailable videos
- Configuration validation test to ensure required environment variables are present.

### Success Metrics
- 100% pass rate on URL parsing test cases covering expected YouTube URL formats.
- Invalid or malformed URLs are rejected before transcript processing.
- Duplicate submissions are detected consistently.
- A user can submit a valid video and receive a deterministic processing status without app failure.

### Exit Criteria
- The app can accept a valid YouTube URL, extract the `video_id`, and hand off a normalized ingestion payload to the transcript layer.
- All intake-related tests pass.

## Phase 2: Transcript Extraction and Normalization

### Goal
Retrieve transcript data in a consistent internal format and handle transcript-related failure cases safely.

### Significant Components
- Integrate `youtube-transcript-api` as the primary transcript source.
- Normalize transcript output to:
  - `text`
  - `start`
  - `duration`
- Implement fallback handling for:
  - missing transcripts
  - disabled transcripts
  - extraction errors
- Add transcript metadata checks such as total segments, empty transcript detection, and total duration coverage.

### Deliverables
- Transcript extraction service module.
- Normalized transcript schema.
- Error-handling paths for videos with no usable transcript.
- Persistence-ready transcript payload for chunking.

### Testing
- Unit tests for transcript normalization.
- Mocked integration tests for:
  - successful transcript fetch
  - missing transcript case
  - empty transcript case
  - extractor failure or timeout
- Manual validation with a small set of videos that cover:
  - short video
  - long video
  - no transcript

### Success Metrics
- Successful transcript extraction for supported videos returns normalized segments with timestamps.
- Transcript normalization produces no empty or malformed entries.
- Unsupported transcript scenarios fail gracefully with a user-readable message.
- No downstream processing begins when transcript extraction fails.

### Exit Criteria
- The system can fetch and normalize transcript data for supported videos and stop cleanly when transcript data is unavailable.
- Transcript tests and failure-path tests pass.

## Phase 3: Sentence-Aware BERT Chunking and Metadata Preparation

### Goal
Convert transcripts into retrieval-optimized chunks using sentence boundaries and BERT token limits while preserving accurate timing metadata.

### Significant Components
- Use `nltk.sent_tokenize` to divide transcript text into sentences.
- Map each sentence to a `start_time` and `end_time`.
- Integrate a HuggingFace BERT tokenizer such as `bert-base-uncased` for token counting.
- Implement chunking logic that:
  - adds sentences sequentially
  - keeps each chunk at `<= 512` BERT tokens
  - never splits a sentence across chunks
  - preserves sentence order and chunk timing continuity
- Generate chunk metadata:
  - `chunk_id`
  - `video_id`
  - `start_time`
  - `end_time`
  - chunk text
- Fail fast when a single sentence exceeds the 512-token BERT limit, because that cannot be chunked without violating the sentence-boundary rule.

### Deliverables
- Reusable chunking engine.
- Structured chunk model for embedding and storage.
- Metadata validation logic for chunk order and timing continuity.

### Testing
- Unit tests for:
  - token counting
  - chunk boundary creation
  - sentence timestamp mapping
  - metadata assignment
- Edge-case tests for:
  - very short transcripts
  - long transcripts
  - transcript segments with sparse text
  - a single sentence exceeding 512 BERT tokens
- Manual review of generated chunks for timing accuracy and contextual continuity.

### Success Metrics
- Chunk sizes never exceed 512 BERT tokens unless processing is explicitly rejected.
- Start and end timestamps align with the underlying transcript segments.
- Retrieval-oriented chunk review shows sentence-complete chunks with no sentence splits at chunk boundaries.

### Exit Criteria
- Transcript inputs consistently produce valid chunk objects ready for embedding and storage.
- Chunking tests pass and chunk samples are manually verified.

## Phase 4: Embeddings, PostgreSQL Storage, and Vector Retrieval

### Goal
Persist chunk embeddings in `PostgreSQL + pgvector` and support reliable semantic retrieval for user queries.

### Significant Components
- Provision PostgreSQL with `pgvector`.
- Create the `video_chunks` table and indexes.
- Implement batch embedding generation using OpenAI embeddings.
- Store embeddings and metadata for each chunk.
- Prevent repeated embedding of already-ingested videos.
- Implement query embedding and similarity search with top-k retrieval.
- Return retrieved chunks with citation metadata.

### Deliverables
- Database schema and migration scripts.
- Embedding pipeline with batching and retry behavior.
- Repository or service layer for insert and retrieval operations.
- Retrieval endpoint or internal function returning ranked chunks.

### Testing
- Schema validation and migration test.
- Integration tests for:
  - chunk insert
  - duplicate video prevention
  - query embedding generation
  - top-k retrieval
- Retrieval quality tests using hand-crafted questions against known videos.
- Performance tests for embedding throughput and retrieval latency on representative data sizes.

### Success Metrics
- 100% of valid chunks for a processed video are stored with embeddings.
- Duplicate ingestion does not create redundant vectors.
- Top-k retrieval returns clearly relevant chunks for benchmark questions.
- Query retrieval latency remains acceptable for MVP usage.

### Exit Criteria
- The system can ingest a video end-to-end into the vector store and retrieve relevant chunks for a query.
- Database and retrieval tests pass.

## Phase 5: Answer Generation and Relevance Detection

### Goal
Generate grounded answers from retrieved context and reject questions that are not answerable from the video content.

### Significant Components
- Implement query-to-answer flow:
  - embed query
  - retrieve top-k chunks
  - run relevance detection
  - generate answer from context only
- Implement relevance detection in two layers:
  - similarity threshold heuristic for MVP
  - optional LLM-based answerability check
- Design prompts that enforce:
  - answer only from provided context
  - no fabrication
  - optional citation or timestamp references
- Return structured output:
  - answer
  - irrelevant-question message when triggered
  - source chunks or timestamps when available

### Deliverables
- Retriever-to-LLM orchestration layer.
- Relevance detection module and threshold configuration.
- Prompt templates for answering and classification.
- Structured response model for the UI.

### Testing
- Unit tests for relevance threshold logic.
- Integration tests for:
  - answerable question flow
  - irrelevant question flow
  - ambiguous question handling
  - empty retrieval result handling
- Manual evaluation set with labeled questions:
  - relevant and easy
  - relevant but multi-hop
  - irrelevant
  - ambiguous
- Prompt regression testing to ensure the model does not answer outside the supplied context.

### Success Metrics
- High percentage of relevant questions produce grounded answers supported by retrieved chunks.
- Irrelevant-question detection correctly blocks unsupported queries at an acceptable rate.
- Hallucination rate is low in manual evaluation.
- Returned answers consistently reference the correct part of the transcript when citations are enabled.

### Exit Criteria
- The system can answer supported questions using retrieved context and refuse unsupported ones reliably enough for MVP use.
- Relevance and answer-generation evaluation gates are met.

## Phase 6: End-to-End UX, Observability, and MVP Release Readiness

### Goal
Turn the working pipeline into a usable MVP with operational visibility, basic safeguards, and deployment readiness.

### Significant Components
- Complete Streamlit chat experience for:
  - processing status
  - question input
  - answer display
  - irrelevant-question messaging
  - optional retrieved chunk display
- Add timestamp-based source references and clickable links where practical.
- Add logging, metrics, and error reporting across the full pipeline.
- Add rate limiting and token-usage safeguards.
- Finalize deployment target:
  - Streamlit Cloud for MVP, or
  - Render or Railway if backend separation is needed
- Document environment setup, runbook, and known limitations.

### Deliverables
- Fully connected Streamlit MVP.
- Basic operational dashboard or logs for failures, latency, and processing outcomes.
- Deployment configuration.
- README or launch documentation for setup, usage, and troubleshooting.

### Testing
- End-to-end tests covering:
  - process a valid video
  - ask a relevant question
  - ask an irrelevant question
  - handle missing transcript case
  - reprocess duplicate video
- Manual usability test of the full user journey.
- Smoke test in deployed environment.
- Latency test for:
  - video processing time
  - question response time

### Success Metrics
- Users can complete the full flow from URL input to answer without developer intervention.
- Processing failures are observable and diagnosable from logs.
- End-to-end success rate is stable for the MVP test set.
- Query response time and video processing time remain within acceptable limits for the intended users.

### Exit Criteria
- The MVP is deployable, testable in a hosted environment, and operationally understandable.
- End-to-end tests and deployment smoke tests pass.

## Phase Transition Rule

Do not move to the next phase until the current phase satisfies all of the following:
- Core deliverables are implemented.
- Phase-specific automated tests pass.
- Manual validation for the phase is completed.
- Success metrics are met or reviewed and explicitly accepted.
- Known blockers for downstream phases are documented.

## Recommended Execution Order Summary

1. Foundation and intake
2. Transcript extraction
3. Token-aware chunking
4. Embeddings and vector retrieval
5. Answer generation and relevance detection
6. UX hardening, observability, and deployment

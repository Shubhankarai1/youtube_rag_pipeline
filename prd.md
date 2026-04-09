Product Requirements Document (PRD)
Product: YouTube RAG Pipeline (End-to-End QA System)

1. Objective
Build an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to input a YouTube video URL and ask questions based on its content. The system should retrieve relevant transcript chunks and generate accurate answers, or detect and respond when a query is irrelevant.

2. Target Users
AI builders and developers learning RAG
Knowledge workers extracting insights from videos
Coaches and educators analyzing long-form content
Internal teams building knowledge retrieval systems

3. Core Features
3.1 Input YouTube Link
User inputs a YouTube URL via UI
Validate:
Proper YouTube format
Video availability
Extract:
video_id
Metadata (title, duration if needed)

3.2 Transcript Extraction
Supported Methods:
Primary: youtube-transcript-api
Fallback: Manual upload or YouTube Data API
Requirements:
Fetch full transcript with timestamps
Handle:
Missing transcripts
Multi-language (optional future scope)
Output Format:
[
  {
    "text": "...",
    "start": 12.5,
    "duration": 4.2
  }
]

3.3 Sentence-Aware BERT Chunking
Goal: Optimize retrieval quality
Approach:
Use NLTK sent_tokenize for sentence segmentation
Use a HuggingFace BERT tokenizer (e.g. bert-base-uncased) for token counting
Chunk size: <= 512 BERT tokens
Logic:
Split transcript text into sentences
Map every sentence to a start and end timestamp
Sequentially add whole sentences to a chunk until adding another sentence would exceed 512 tokens
Never split a sentence across chunks
Maintain:
chunk_id
start_time
end_time
Output:
{
  "chunk_id": "vid123_01",
  "text": "...",
  "start_time": 10,
  "end_time": 45
}

3.4 Embeddings + Storage (PostgreSQL + pgvector)
Embedding Model:
OpenAI embeddings (e.g. text-embedding-3-small)
Database:
PostgreSQL with pgvector
Schema:
Table: video_chunks

- id (UUID)
- video_id (TEXT)
- chunk_id (TEXT)
- content (TEXT)
- embedding (VECTOR)
- start_time (FLOAT)
- end_time (FLOAT)
- created_at (TIMESTAMP)

Requirements:
Batch embedding for efficiency
Store metadata for retrieval and citations

3.5 Query Answering + Relevance Detection
Step 1: Query Embedding
Convert user query to embedding
Step 2: Retrieval
Perform similarity search (top-k = 3-5)
Use cosine similarity
Step 3: Relevance Filter
Two approaches:
Option A (fast heuristic):
If similarity score < threshold, mark as irrelevant
Option B (better):
LLM classification:
"Is this question answerable from the given context?"
Step 4: Response Generation
Pass retrieved chunks into LLM
Prompt structure:
Context + Question
Instruction: "Answer only from context"
Step 5: Output
Answer
Optional:
Timestamp references
Source chunks

4. UI (Streamlit)
Main Components:
1. Input Section
YouTube URL field
"Process Video" button
2. Status Indicators
Transcript extraction
Chunking
Embedding progress
3. Chat Interface
Input: User question
Output:
Answer
"Irrelevant question" message (if triggered)
4. Optional Enhancements
Show retrieved chunks
Clickable timestamps to open video

5. System Architecture
[Streamlit UI]
      down
[Backend API / Logic Layer]
      down
1. Transcript Extractor
2. Chunking Engine (NLTK + HuggingFace BERT tokenizer)
3. Embedding Generator (OpenAI)
4. Vector Store (PostgreSQL + pgvector)
      down
[Retriever + LLM Layer]
      down
[Response]

6. Deployment
MVP Deployment Options:
Streamlit Cloud (fastest)
Render / Railway
Local + ngrok (testing)
Production Setup:
Backend: FastAPI
DB: Managed PostgreSQL (Supabase / Neon)
Queue (optional): Celery / Redis for async processing

7. Success Metrics
Retrieval accuracy (relevant chunk match)
Answer correctness (manual eval or LLM eval)
Latency:
Processing time per video
Query response time
Percent of correctly flagged irrelevant queries

8. Edge Cases
No transcript available
Very long videos (token explosion)
Irrelevant queries
Ambiguous questions
Duplicate ingestion of same video

9. Security & Limits
Rate limit API calls
Handle OpenAI token usage carefully
Prevent repeated embeddings for same video

10. Future Enhancements
Multi-video knowledge base
Semantic search across videos
Auto-summary generation
Highlight clips based on queries
Speaker detection
Fine-tuned relevance classifier

11. Tech Stack
Frontend: Streamlit
Backend: Python
LLM: OpenAI
Embeddings: OpenAI
DB: PostgreSQL + pgvector
Chunking: NLTK + HuggingFace BERT tokenizer
Transcript: youtube-transcript-api

12. Key Design Principles
Retrieval quality > fancy UI
Metadata = huge leverage (timestamps)
Always validate relevance before answering
Never split a sentence across chunk boundaries

13. Advanced RAG: Continuous Knowledge Chat
Goal:
Turn the app from a single-video Q&A tool into a continuous knowledge assistant where users can chat with everything they have uploaded so far.

13.1 Core Product Idea
The system should behave like one growing brain
Every uploaded video or document becomes part of that brain
Users should be able to chat continuously without resetting or reprocessing previous uploads

13.2 Chat Experience
By default, the user chats with all available knowledge
The system retrieves context from any previously uploaded and indexed content
When a new video or document is added, it becomes available in the same chat experience after processing completes

13.3 User Control
Provide optional scope control with two modes:
All Content (default) -> search everything
Selected Content -> user chooses specific videos or documents
This control should remain simple and non-intrusive

13.4 Upload Behavior
When a user adds a new video or document:
It gets validated and processed
It is added to the existing knowledge base
The active chat continues seamlessly without reset

13.5 Retrieval Requirements
The retrieval layer must support:
Searching across all indexed content
Searching within a selected subset of sources
Answers must stay relevant as the corpus grows
If multiple sources are used in one answer, that should be made clear to the user
Returned chunks should include source metadata for attribution and debugging

13.6 Deduplication and Persistence
Avoid reprocessing the same video or document multiple times
Deduplication should be persistent, not only session-based
Each source should have a stable identity in storage
Indexed sources should remain available across sessions

13.7 Data Model Direction
The architecture should evolve from video-centric storage to source-centric storage
Recommended logical entities:

Source
- source_id
- source_type (youtube, document, etc.)
- external_id or file hash
- title
- processing_status
- created_at

Chunk
- chunk_id
- source_id
- content
- embedding
- timestamp or page reference
- created_at

Chat Scope
- mode: all_content or selected_content
- selected_source_ids

13.8 Expected Outcome
The app should feel closer to ChatGPT with memory
Users can build a personal knowledge base over time
There should be no need to re-upload content or switch contexts repeatedly
The product should stay simple for beginners and flexible for advanced users

13.9 Implementation Principles for Advanced RAG
Keep chat memory separate from knowledge-base memory
Prefer persistent source-level deduplication over temporary in-memory checks
Make source attribution explicit when answers rely on multiple uploads
Preserve the default simplicity of the chat UX even as retrieval becomes multi-source

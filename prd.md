📄 Product Requirements Document (PRD)
Product: YouTube RAG Pipeline (End-to-End QA System)

1. 🎯 Objective
Build an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to input a YouTube video URL and ask questions based on its content. The system should retrieve relevant transcript chunks and generate accurate answers, or detect and respond when a query is irrelevant.

2. 👤 Target Users
AI builders / developers learning RAG
Knowledge workers extracting insights from videos
Coaches / educators analyzing long-form content
Internal teams building knowledge retrieval systems

3. 🧠 Core Features
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
Fallback: Manual upload / YouTube Data API
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

3.3 Token-Aware Chunking
Goal: Optimize retrieval quality
Approach:
Use tiktoken for token counting
Chunk size: ~300–500 tokens
Overlap: 10–20% (critical for context continuity)
Logic:
Merge transcript lines into chunks
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
Store metadata for retrieval + citations

3.5 Query Answering + Relevance Detection
Step 1: Query Embedding
Convert user query → embedding
Step 2: Retrieval
Perform similarity search (top-k = 3–5)
Use cosine similarity
Step 3: Relevance Filter (IMPORTANT)
Two approaches:
Option A (Fast heuristic):
If similarity score < threshold → irrelevant
Option B (Better):
LLM classification:
“Is this question answerable from the given context?”
Step 4: Response Generation
Pass retrieved chunks into LLM
Prompt structure:
Context + Question
Instruction: “Answer ONLY from context”
Step 5: Output
Answer
Optional:
Timestamp references
Source chunks

4. 🖥️ UI (Streamlit)
Main Components:
1. Input Section
YouTube URL field
“Process Video” button
2. Status Indicators
Transcript extraction
Chunking
Embedding progress
3. Chat Interface
Input: User question
Output:
Answer
“Irrelevant question” message (if triggered)
4. Optional Enhancements
Show retrieved chunks
Clickable timestamps → open video

5. ⚙️ System Architecture
[Streamlit UI]
      ↓
[Backend API / Logic Layer]
      ↓
1. Transcript Extractor
2. Chunking Engine (tiktoken)
3. Embedding Generator (OpenAI)
4. Vector Store (PostgreSQL + pgvector)
      ↓
[Retriever + LLM Layer]
      ↓
[Response]

6. 🚀 Deployment
MVP Deployment Options:
Streamlit Cloud (fastest)
Render / Railway
Local + ngrok (testing)
Production Setup:
Backend: FastAPI
DB: Managed PostgreSQL (Supabase / Neon)
Queue (optional): Celery / Redis for async processing

7. 📊 Success Metrics
Retrieval accuracy (relevant chunk match)
Answer correctness (manual eval or LLM eval)
Latency:
Processing time per video
Query response time
% of correctly flagged irrelevant queries

8. ⚠️ Edge Cases
No transcript available
Very long videos (token explosion)
Irrelevant queries
Ambiguous questions
Duplicate ingestion of same video

9. 🔐 Security & Limits
Rate limit API calls
Handle OpenAI token usage carefully
Prevent repeated embeddings for same video

10. 🔮 Future Enhancements
Multi-video knowledge base
Semantic search across videos
Auto-summary generation
Highlight clips based on queries
Speaker detection
Fine-tuned relevance classifier

11. 🧩 Tech Stack
Frontend: Streamlit
Backend: Python
LLM: OpenAI
Embeddings: OpenAI
DB: PostgreSQL + pgvector
Chunking: tiktoken
Transcript: youtube-transcript-api

12. 💡 Key Design Principles
Retrieval quality > fancy UI
Overlap chunking is non-negotiable
Metadata = huge leverage (timestamps)
Always validate relevance before answering 


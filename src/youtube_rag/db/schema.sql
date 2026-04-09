CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    source_url TEXT,
    normalized_url TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_type_external_id
    ON sources (source_type, external_id);

CREATE TABLE IF NOT EXISTS video_chunks (
    id UUID PRIMARY KEY,
    video_id TEXT NOT NULL,
    source_id TEXT,
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_chunks_video_id
    ON video_chunks (video_id);

CREATE INDEX IF NOT EXISTS idx_video_chunks_source_id
    ON video_chunks (source_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_video_chunks_chunk_id
    ON video_chunks (chunk_id);

CREATE INDEX IF NOT EXISTS idx_video_chunks_embedding_cosine
    ON video_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

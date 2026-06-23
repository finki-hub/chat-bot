CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS question (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    user_id TEXT,
    links JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS question_name_idx ON question (name);

CREATE TABLE IF NOT EXISTS link (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    description TEXT,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS link_name_idx ON link (name);

-- Embeddings

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_llama3_3_70b vector (8192);

-- No indexing for llama3_3_70b because indexes support up to 2000 dimensions

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector (1024);

CREATE INDEX IF NOT EXISTS question_embedding_bge_m3_idx ON question USING hnsw (
    embedding_bge_m3 vector_cosine_ops
);

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_text_embedding_3_large vector (3072);

-- vector HNSW supports up to 2000 dims, but halfvec supports up to 4000 dims
CREATE INDEX IF NOT EXISTS question_embedding_text_embedding_3_large_idx ON question USING hnsw (
    (embedding_text_embedding_3_large::halfvec(3072)) halfvec_cosine_ops
);

ALTER TABLE question
DROP COLUMN IF EXISTS embedding_text_embedding_004;

ALTER TABLE question
DROP COLUMN IF EXISTS embedding_text_embedding_005;

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_gemini_embedding_001 vector (3072);

-- vector HNSW supports up to 2000 dims, but halfvec supports up to 4000 dims
CREATE INDEX IF NOT EXISTS question_embedding_gemini_embedding_001_idx ON question USING hnsw (
    (embedding_gemini_embedding_001::halfvec(3072)) halfvec_cosine_ops
);

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_multilingual_e5_large vector (1024);

CREATE INDEX IF NOT EXISTS question_embedding_multilingual_e5_large_idx ON question USING hnsw (
    embedding_multilingual_e5_large vector_cosine_ops
);

-- Source-of-truth documents and their chunks (a second retrieval unit, searched alongside question).

CREATE TABLE IF NOT EXISTS document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    source_type TEXT,
    source_hash TEXT,
    metadata JSONB,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS document_name_idx ON document (name);

CREATE TABLE IF NOT EXISTS chunk (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    document_id UUID NOT NULL REFERENCES document (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    section TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS chunk_document_chunk_idx ON chunk (document_id, chunk_index);

CREATE INDEX IF NOT EXISTS chunk_document_id_idx ON chunk (document_id);

-- Chunk embedding columns mirror `question` (same names/dims), reused by the MODEL_* maps.

ALTER TABLE chunk
ADD COLUMN IF NOT EXISTS embedding_llama3_3_70b vector (8192);

-- No indexing for llama3_3_70b because indexes support up to 2000 dimensions

ALTER TABLE chunk
ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector (1024);

CREATE INDEX IF NOT EXISTS chunk_embedding_bge_m3_idx ON chunk USING hnsw (
    embedding_bge_m3 vector_cosine_ops
);

ALTER TABLE chunk
ADD COLUMN IF NOT EXISTS embedding_text_embedding_3_large vector (3072);

-- vector HNSW supports up to 2000 dims, but halfvec supports up to 4000 dims
CREATE INDEX IF NOT EXISTS chunk_embedding_text_embedding_3_large_idx ON chunk USING hnsw (
    (embedding_text_embedding_3_large::halfvec(3072)) halfvec_cosine_ops
);

ALTER TABLE chunk
ADD COLUMN IF NOT EXISTS embedding_gemini_embedding_001 vector (3072);

-- vector HNSW supports up to 2000 dims, but halfvec supports up to 4000 dims
CREATE INDEX IF NOT EXISTS chunk_embedding_gemini_embedding_001_idx ON chunk USING hnsw (
    (embedding_gemini_embedding_001::halfvec(3072)) halfvec_cosine_ops
);

ALTER TABLE chunk
ADD COLUMN IF NOT EXISTS embedding_multilingual_e5_large vector (1024);

CREATE INDEX IF NOT EXISTS chunk_embedding_multilingual_e5_large_idx ON chunk USING hnsw (
    embedding_multilingual_e5_large vector_cosine_ops
);

-- Response feedback (like/dislike). Consumer-agnostic, keyed by the response_id the
-- chatbot mints per /chat and returns via the X-Response-Id header. question/answer/model
-- columns are client-attested (the server stores what the consumer reports, unverified).

CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    response_id UUID NOT NULL,
    client TEXT NOT NULL CHECK (client IN ('discord', 'web')),
    user_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('like', 'dislike')),
    client_ref TEXT,
    channel_id TEXT,
    guild_id TEXT,
    question_text TEXT,
    answer_text TEXT,
    inference_model TEXT,
    embeddings_model TEXT,
    query_transform_model TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS feedback_response_client_user_idx ON feedback (response_id, client, user_id);

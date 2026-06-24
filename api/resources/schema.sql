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

-- Diploma defenses (committee corpus). One row = one defense = the retrieval unit.
-- Committee members are stored as canonical-name TEXT; their unordered nature is
-- enforced in scoring (frozenset), not DDL.

CREATE TABLE IF NOT EXISTS diploma (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    external_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    mentor TEXT NOT NULL,
    member1 TEXT,
    member2 TEXT,
    status TEXT NOT NULL,
    date_of_submission DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS diploma_external_id_idx ON diploma (external_id);

CREATE INDEX IF NOT EXISTS diploma_mentor_idx ON diploma (mentor);

-- Diploma embedding columns mirror `question`/`chunk` (same names/dims), reused by the MODEL_* maps.

ALTER TABLE diploma
ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector (1024);

CREATE INDEX IF NOT EXISTS diploma_embedding_bge_m3_idx ON diploma USING hnsw (
    embedding_bge_m3 vector_cosine_ops
);

-- Professor publications (paper corpus for the expertise + co-author "buddy" signals).
-- canonical_authors hold the SAME canonical-name strings as diploma mentor/member, so the
-- paper signals compose with the defense graph without a name-mapping layer at query time.

CREATE TABLE IF NOT EXISTS professor_document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    external_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    topics JSONB,
    canonical_authors JSONB,
    sources JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS professor_document_external_id_idx ON professor_document (external_id);

ALTER TABLE professor_document
ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector (1024);

CREATE INDEX IF NOT EXISTS professor_document_embedding_bge_m3_idx ON professor_document USING hnsw (
    embedding_bge_m3 vector_cosine_ops
);

-- Temporal staff groups: clusters of professors who repeatedly served/published together
-- within a time window (community detection over the co-occurrence graph, per window).
-- Recomputed wholesale per source by compute_professor_groups.

CREATE TABLE IF NOT EXISTS professor_group (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    source TEXT NOT NULL,
    window_start INTEGER NOT NULL,
    window_end INTEGER NOT NULL,
    group_index INTEGER NOT NULL,
    members JSONB NOT NULL,
    size INTEGER NOT NULL,
    min_weight INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS professor_group_window_idx ON professor_group (
    source, window_start, window_end, group_index
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

-- Answers whose knowledge-base searches found nothing usable. No question text:
-- join on response_id to chat_message while the conversation still exists.
CREATE TABLE IF NOT EXISTS retrieval_gap (
    response_id UUID PRIMARY KEY,
    interface TEXT NOT NULL CHECK (interface IN ('discord', 'web')),
    reason TEXT NOT NULL CHECK (reason IN ('no_results', 'weak_results')),
    -- The agent's own keyword queries; NULL entries are searches on the question itself.
    search_queries TEXT[] NOT NULL,
    inference_model TEXT NOT NULL,
    embeddings_model TEXT NOT NULL,
    reranker_score_max REAL,
    outcome TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS retrieval_gap_created_at_idx ON retrieval_gap (created_at DESC);

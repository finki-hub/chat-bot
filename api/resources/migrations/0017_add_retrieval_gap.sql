-- Answers whose knowledge-base searches found nothing usable. The question itself
-- is not stored: join on response_id to chat_message. The agent's queries are
-- derived from it, so rows are deleted with their conversation, and rows whose
-- answer was never stored or was later replaced are swept after an hour.
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

CREATE INDEX IF NOT EXISTS chat_message_assistant_response_id_idx ON chat_message (response_id)
WHERE role = 'assistant' AND response_id IS NOT NULL;

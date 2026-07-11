CREATE TABLE IF NOT EXISTS chat_user_credential (
    user_id UUID NOT NULL REFERENCES chat_user (id) ON DELETE CASCADE,
    provider TEXT NOT NULL CHECK (provider IN ('openai', 'google', 'anthropic', 'ollama')),
    encrypted_api_key TEXT NOT NULL,
    base_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, provider)
);

CREATE INDEX IF NOT EXISTS chat_user_credential_user_updated_idx ON chat_user_credential (user_id, updated_at DESC);

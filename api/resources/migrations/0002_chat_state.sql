CREATE TABLE IF NOT EXISTS chat_user (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    provider TEXT NOT NULL CHECK (provider IN ('google')),
    provider_subject TEXT NOT NULL,
    email TEXT,
    name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (provider, provider_subject)
);

CREATE TABLE IF NOT EXISTS chat_conversation (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES chat_user (id) ON DELETE CASCADE,
    active_stream_id UUID,
    active_response_id UUID,
    active_status TEXT CHECK (active_status IN ('pending', 'streaming', 'completed', 'stopped', 'error')),
    model TEXT,
    title TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_conversation_user_updated_idx ON chat_conversation (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_message (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES chat_conversation (id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    response_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_message_conversation_created_idx ON chat_message (conversation_id, created_at);

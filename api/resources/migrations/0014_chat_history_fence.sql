-- Account epochs fence delayed writes across workers and survive empty-history deletion.
ALTER TABLE chat_user ADD COLUMN history_generation BIGINT NOT NULL DEFAULT 0;

-- A deleted conversation UUID must not be recreated by a delayed Discord exchange.
-- No transcript content is retained here; a new conversation uses a new UUID.
CREATE TABLE chat_conversation_tombstone (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES chat_user (id) ON DELETE CASCADE
);

CREATE INDEX chat_conversation_tombstone_user_idx ON chat_conversation_tombstone (user_id);
CREATE INDEX chat_message_discord_response_idx ON chat_message (response_id)
WHERE role = 'assistant' AND metadata ->> 'interface' = 'discord';

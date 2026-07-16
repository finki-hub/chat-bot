ALTER TABLE chat_conversation
ADD COLUMN IF NOT EXISTS share_token UUID;

CREATE UNIQUE INDEX IF NOT EXISTS chat_conversation_share_token_idx
ON chat_conversation (share_token)
WHERE share_token IS NOT NULL;

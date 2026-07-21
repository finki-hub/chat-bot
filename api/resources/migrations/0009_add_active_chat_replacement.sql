ALTER TABLE chat_conversation
ADD COLUMN IF NOT EXISTS active_replacement_message_id UUID;

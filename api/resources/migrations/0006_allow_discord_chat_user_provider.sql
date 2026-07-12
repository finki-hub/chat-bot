ALTER TABLE chat_user
DROP CONSTRAINT IF EXISTS chat_user_provider_check;

ALTER TABLE chat_user
ADD CONSTRAINT chat_user_provider_check CHECK (provider IN ('google', 'microsoft-entra-id', 'discord'));

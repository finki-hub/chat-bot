ALTER TABLE chat_user_credential
DROP CONSTRAINT IF EXISTS chat_user_credential_provider_check;

ALTER TABLE chat_user_credential
ADD CONSTRAINT chat_user_credential_provider_check
CHECK (provider IN ('openai', 'google', 'anthropic', 'ollama', 'openrouter'));

import type { UIMessage } from 'ai';

export type ChatRequestBody = {
  embeddings_model?: ModelId; // default BAAI/bge-m3
  inference_model?: ModelId; // default claude-sonnet-4-6
  max_tokens?: number; // >= 1 (default 4096)
  messages: ConversationTurn[]; // 1..50, oldest-first, last is role:"user"
  query_transform_model?: ModelId; // default gpt-5.4-mini
  system_prompt?: null | string; // default null -> server FINKI agent prompt
  temperature?: number; // 0.0..1.0 (default 0.3)
  top_p?: number; // 0.0..1.0 (default 1.0)
};

export type ConversationRole = 'assistant' | 'user';

export type ConversationTurn = {
  content: string; // <= 8000 chars
  role: ConversationRole;
};

// eslint-disable-next-line sonarjs/redundant-type-aliases -- semantic alias for the public wire contract; consumers reference ModelId, not bare string
export type ModelId = string;

export const MAX_MESSAGES = 50;
export const MAX_CHARS_PER_TURN = 8_000;

export type ChatErrorCode = 'agent_error' | 'interrupted' | 'no_answer';

export type FeedbackAck = {
  feedback_type: FeedbackType;
  id: string; // UUID, not an int
  response_id: string;
};

// The BFF adds client:"web", user_id, x-api-key.
export type FeedbackClientPayload = {
  answerText?: string;
  feedbackType: FeedbackType;
  inferenceModel?: string;
  questionText?: string;
  responseId: string;
  userId: string; // anon per-browser id (BFF maps to user_id)
};

export type FeedbackSchema = {
  answer_text?: string;
  channel_id?: string;
  client: 'web';
  client_ref?: string;
  embeddings_model?: string;
  feedback_type: FeedbackType;
  guild_id?: string;
  inference_model?: string;
  query_transform_model?: string;
  question_text?: string;
  response_id: string; // UUID from X-Response-Id
  user_id: string; // required, min length 1 (anon per-browser id)
};

export type FeedbackType = 'dislike' | 'like';

export type MyDataParts = {
  error: { code: string; message: string };
  status: { label: string; tool?: string };
};

export type MyMetadata = {
  inferenceModel?: string;
  responseId?: string;
};

export type MyUIMessage = UIMessage<MyMetadata, MyDataParts>;

export type ProtocolV2Event =
  | { data: Record<string, never>; event: 'done' }
  | { data: Record<string, never>; event: 'reset' }
  | {
      data: {
        code: 'agent_error' | 'interrupted' | 'no_answer';
        message: string;
      };
      event: 'error';
    }
  | { data: { label: string; state: string; tool?: string }; event: 'status' }
  | { data: { text: string }; event: 'token' };

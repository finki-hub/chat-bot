// Single source of truth for the wire contract + the typed UIMessage.
// Mirrors the Python API: ChatSchema (api/app/schemas/chat.py) and
// FeedbackSchema (api/app/schemas/feedback.py), per spec §3.2 / §3.4.
import type { UIMessage } from 'ai';

export type ChatRequestBody = {
  embeddings_model?: ModelId; // default BAAI/bge-m3
  inference_model?: ModelId; // default claude-sonnet-4-6 — picks streaming LLM
  max_tokens?: number; // >= 1 (default 4096)
  messages: ConversationTurn[]; // 1..50, oldest-first, last is role:"user"
  query_transform_model?: ModelId; // default gpt-5.4-mini
  system_prompt?: null | string; // default null -> server FINKI agent prompt
  temperature?: number; // 0.0..1.0 (default 0.3)
  top_p?: number; // 0.0..1.0 (default 1.0)
};

// ---------------------------------------------------------------------------
// POST /chat/ request body (ChatSchema). messages: 1..50, OLDEST-FIRST, last
// element MUST be role:"user". content <= 8000 chars/turn. snake_case on wire.
// ---------------------------------------------------------------------------
export type ConversationRole = 'assistant' | 'user';

export type ConversationTurn = {
  content: string; // <= 8000 chars
  role: ConversationRole;
};

// ---------------------------------------------------------------------------
// Model ids: GET /chat/models returns a flat sorted string[]; no display meta.
// A semantic alias for documentation; the wire value is an opaque model string.
// ---------------------------------------------------------------------------
// eslint-disable-next-line sonarjs/redundant-type-aliases -- semantic alias for the public wire contract; consumers reference ModelId, not bare string
export type ModelId = string;

// Client-side caps the BFF re-validates.
export const MAX_MESSAGES = 50;
export const MAX_CHARS_PER_TURN = 8_000;

export type ChatErrorCode = 'agent_error' | 'interrupted' | 'no_answer';

export type FeedbackAck = {
  feedback_type: FeedbackType;
  id: string; // UUID — FeedbackAckSchema.id is a UUID, not an int
  response_id: string; // UUID
};

// Client -> BFF feedback payload; the BFF adds client:"web", user_id, x-api-key.
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
  client: 'web'; // required literal for this app
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

// ---------------------------------------------------------------------------
// POST /chat/feedback (FeedbackSchema). client is the literal "web"; user_id
// is required (>=1 char). x-api-key is injected server-side by the BFF.
// ---------------------------------------------------------------------------
export type FeedbackType = 'dislike' | 'like';

export type MyDataParts = {
  error: { code: string; message: string }; // -> part type "data-error"
  status: { label: string; tool?: string }; // -> part type "data-status"
};

// ---------------------------------------------------------------------------
// Typed UIMessage: metadata + custom data parts. UIMessage<METADATA, DATA, TOOLS>.
// Data parts are named EXACTLY data-status and data-error; both transient.
// ---------------------------------------------------------------------------
export type MyMetadata = {
  inferenceModel?: string;
  responseId?: string;
};

export type MyUIMessage = UIMessage<MyMetadata, MyDataParts>;

// ---------------------------------------------------------------------------
// protocol-v2 SSE events (named SSE: `event: <name>\ndata: <JSON>\n\n`).
// ---------------------------------------------------------------------------
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

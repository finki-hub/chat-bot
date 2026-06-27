import type { UIMessage } from 'ai';

export type ChatRequestBody = {
  embeddings_model?: ModelId;
  inference_model?: ModelId;
  max_tokens?: number;
  messages: ConversationTurn[];
  query_transform_model?: ModelId;
  reasoning?: boolean;
  system_prompt?: null | string;
  temperature?: number;
  top_p?: number;
};

export type ConversationRole = 'assistant' | 'user';

export type ConversationTurn = {
  content: string;
  role: ConversationRole;
};

// eslint-disable-next-line sonarjs/redundant-type-aliases -- semantic alias for the public wire contract; consumers reference ModelId, not bare string
export type ModelId = string;

export const MAX_MESSAGES = 50;
export const MAX_CHARS_PER_TURN = 8_000;

export type ChatErrorCode = 'agent_error' | 'interrupted' | 'no_answer';

export type FeedbackAck = {
  feedback_type: FeedbackType;
  id: string;
  response_id: string;
};

export type FeedbackClientPayload = {
  answerText?: string;
  feedbackType: FeedbackType;
  inferenceModel?: string;
  questionText?: string;
  responseId: string;
  userId: string;
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
  response_id: string;
  user_id: string;
};

export type FeedbackType = 'dislike' | 'like';

// Every field is optional: a `meta` frame carries only part of it (timing vs tokens),
// and not all providers report token usage.
export type MessageDiagnostics = {
  candidateCount?: null | number;
  serverTotalMs?: null | number;
  serverTtftMs?: null | number;
  spans?: Record<string, number>;
  // Wall-clock before the answer: first thinking frame → first token.
  thinkingMs?: null | number;
  tokens?: { input: number; output: number; total: number };
  topDistance?: null | number;
};

export type MyDataParts = {
  error: { code: string; message: string };
  status: { label: string; tool?: string };
};

export type MyMetadata = {
  diagnostics?: MessageDiagnostics;
  feedback?: FeedbackType;
  inferenceModel?: string;
  responseId?: string;
  timing?: { totalMs: number; ttftMs: null | number };
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
  | { data: { text: string }; event: 'thinking' }
  | { data: { text: string }; event: 'token' }
  | {
      data: {
        timing?: {
          candidate_count: null | number;
          spans: Record<string, number>;
          // Optional: an older server (pre-rollout) won't emit it.
          thinking_ms?: null | number;
          top_distance: null | number;
          total_ms: null | number;
          ttft_ms: null | number;
        };
        tokens?: { input: number; output: number; total: number };
      };
      event: 'meta';
    };

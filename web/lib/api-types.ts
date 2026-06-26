import type { UIMessage } from 'ai';

export type ChatRequestBody = {
  embeddings_model?: ModelId;
  inference_model?: ModelId;
  max_tokens?: number;
  messages: ConversationTurn[];
  query_transform_model?: ModelId;
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

export type MyDataParts = {
  error: { code: string; message: string };
  status: { label: string; tool?: string };
};

export type MyMetadata = {
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
  | { data: { text: string }; event: 'token' };

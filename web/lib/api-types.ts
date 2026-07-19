import type { UIMessage } from 'ai';

export type CatalogProvider = 'anthropic' | 'google' | 'ollama' | 'openai';

export type CatalogSource = 'error' | 'live' | 'snapshot' | 'stale';

export type ChatCredentialProvider =
  | 'anthropic'
  | 'google'
  | 'ollama'
  | 'openai';

export type ChatCredentialPublic = {
  readonly base_url: null | string;
  readonly created_at?: string;
  readonly has_api_key: boolean;
  readonly provider: ChatCredentialProvider;
  readonly updated_at?: string;
  readonly user_id: string;
};

export type ChatCredentialUpsert = {
  readonly apiKey: string;
  readonly baseUrl?: null | string;
  readonly provider: ChatCredentialProvider;
};

export type ChatInterface = 'discord' | 'web';

export type ChatRequestBody = {
  embeddings_model?: ModelId;
  inference_model?: ModelId;
  interface: ChatInterface;
  max_tokens?: number;
  messages: readonly ConversationTurn[];
  query_transform_mode?: QueryTransformMode;
  query_transform_model?: ModelId;
  reasoning?: boolean;
  temperature?: number;
  top_p?: number;
};

export type ChatTitleClientPayload = {
  readonly messages: readonly ConversationTurn[];
  readonly providerModel?: ModelId;
  readonly queryTransformModel?: ModelId;
};

export type ChatTitleRequestBody = {
  readonly messages: readonly ConversationTurn[];
  readonly provider_model?: ModelId;
  readonly query_transform_model?: ModelId;
  readonly user_id?: string;
};

export type ChatTitleResponse = {
  readonly title: string;
};

export type ConversationRole = 'assistant' | 'user';

export type ConversationTurn = {
  content: string;
  role: ConversationRole;
};

export type ModelAvailability = 'both' | 'byok' | 'sponsored' | 'unavailable';

export type ModelCatalog = {
  readonly models: readonly ModelDescriptor[];
  readonly source: CatalogSource;
  readonly version: 1;
};

// `provider` is a `CatalogProvider` for typed catalogs; for legacy string[] responses
// it is a best-effort bucket inferred from the id, so it stays a plain string.
export type ModelDescriptor = {
  readonly availability?: ModelAvailability;
  readonly description?: string;
  readonly id: ModelId;
  readonly loaded?: boolean | null;
  readonly name: string;
  readonly provider: string;
  readonly sponsored_quota?: SponsoredQuota;
};

// eslint-disable-next-line sonarjs/redundant-type-aliases -- semantic alias for the public wire contract; consumers reference ModelId, not bare string
export type ModelId = string;

export type QueryTransformMode = 'hyde' | 'raw' | 'rewrite' | 'rewrite_hyde';

export type SponsoredQuota = {
  readonly limit: number;
  readonly remaining: number;
  readonly resets_at: string;
};

export const MAX_MESSAGES = 10;
export const MAX_CHARS_PER_TURN = 2_000;

export type ChatErrorCode =
  | 'agent_error'
  | 'credential_required'
  | 'free_quota_exhausted'
  | 'free_tier_unavailable'
  | 'interrupted'
  | 'no_answer'
  | 'sponsored_request_in_progress';

// A user-facing error: the live transient `error` part, the persisted metadata, and
// the in-memory active error all share this shape (`code` is looser than ChatErrorCode
// because it also carries transport codes like 'network'/'pre_stream').
export type ErrorNotice = {
  code: string;
  message: string;
  resets_at?: string;
};

export type FeedbackAck = {
  feedback_type: FeedbackType;
  id: string;
  response_id: string;
};

export type FeedbackClientPayload = {
  readonly feedbackType: FeedbackType;
  readonly responseId: string;
};

export type FeedbackRetractionAck = {
  readonly feedback_type: null;
  readonly response_id: string;
};

export type FeedbackRetractionClientPayload = {
  readonly responseId: string;
};

export type FeedbackRetractionSchema = {
  readonly client: 'web';
  readonly response_id: string;
  readonly user_id: string;
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
export type FeedbackSelection = FeedbackType | null;
export type FeedbackType = 'dislike' | 'like';

// Every field is optional: a `meta` frame carries only part of it (timing vs tokens),
// and not all providers report token usage.
export type MessageDiagnostics = {
  candidateCount?: null | number;
  cost?: { inputUsd: number; outputUsd: number; totalUsd: number };
  serverTotalMs?: null | number;
  serverTtftMs?: null | number;
  spans?: Record<string, number>;
  // Wall-clock before the answer: first thinking frame → first token.
  thinkingMs?: null | number;
  tokens?: { input: number; output: number; total: number };
  topDistance?: null | number;
};

export type MyDataParts = {
  error: ErrorNotice;
  reset: Record<string, never>;
  status: StatusPart;
};

export type MyMetadata = {
  diagnostics?: MessageDiagnostics;
  // Persisted so the notice survives a refresh; the live `error` part is transient.
  error?: ErrorNotice;
  feedback?: FeedbackType;
  inferenceModel?: string;
  responseId?: string;
  sources?: readonly RetrievedSource[];
  timing?: { totalMs: number; ttftMs: null | number };
};

export type MyUIMessage = UIMessage<MyMetadata, MyDataParts>;

export type ProtocolV2Event =
  | { data: Record<string, never>; event: 'done' }
  | { data: Record<string, never>; event: 'reset' }
  | {
      data: {
        code: ChatErrorCode;
        message: string;
        resets_at?: string;
      };
      event: 'error';
    }
  | {
      data: {
        cost?: { input_usd: number; output_usd: number; total_usd: number };
        timing?: {
          candidate_count: null | number;
          spans: Record<string, number>;
          thinking_ms: null | number;
          top_distance: null | number;
          total_ms: null | number;
          ttft_ms: null | number;
        };
        tokens?: { input: number; output: number; total: number };
      };
      event: 'meta';
    }
  | {
      data: { label: string; stage?: string; state: string; tool?: string };
      event: 'status';
    }
  | {
      data: { sources: readonly ProtocolV2RetrievedSource[] };
      event: 'sources';
    }
  | { data: { text: string }; event: 'thinking' }
  | { data: { text: string }; event: 'token' };

export type ProtocolV2RetrievedSource = {
  readonly chunk_index?: number;
  readonly id: string;
  readonly kind: RetrievedSourceKind;
  readonly links?: readonly RetrievedSourceLink[];
  readonly section?: string;
  readonly snippet?: string;
  readonly title: string;
};

export type RetrievedSource = {
  readonly chunkIndex?: number;
  readonly id: string;
  readonly kind: RetrievedSourceKind;
  readonly links?: readonly RetrievedSourceLink[];
  readonly section?: string;
  readonly snippet?: string;
  readonly title: string;
};

export type RetrievedSourceKind = 'chunk' | 'faq';

export type RetrievedSourceLink = {
  readonly label: string;
  readonly url: string;
};

export type StatusPart = { label: string; stage?: string; tool?: string };

import { DefaultChatTransport } from 'ai';
import { posthog } from 'posthog-js';

import type { ModelId, MyUIMessage, QueryTransformMode } from '@/lib/api-types';
import type {
  ChatConversationHistory,
  ConversationRow,
} from '@/lib/conversation-types';

export type ChatExtras = {
  embeddingsModel?: ModelId;
  maxTokens?: number;
  model: ModelId;
  queryTransformMode?: QueryTransformMode;
  queryTransformModel?: ModelId;
  reasoning?: boolean;
  temperature?: number;
  topP?: number;
};

export type StopChatStreamOptions = {
  readonly activeStreamId?: string;
};

export const buildChatTransport = (
  getExtras: () => ChatExtras,
): DefaultChatTransport<MyUIMessage> =>
  new DefaultChatTransport<MyUIMessage>({
    api: '/api/chat',
    prepareReconnectToStreamRequest: ({ id }) => ({
      api: `/api/chat/${encodeURIComponent(id)}/stream`,
    }),
    prepareSendMessagesRequest: ({ id, messageId, messages, trigger }) => ({
      body: {
        id,
        messageId,
        messages,
        trigger,
        ...getExtras(),
        posthogDistinctId: posthog.get_distinct_id(),
        posthogSessionId: posthog.get_session_id(),
      },
    }),
  });

export class ChatConversationRequestError extends Error {
  readonly status: number;

  constructor(status: number, options?: ErrorOptions) {
    super('Chat conversation request failed', options);
    this.name = 'ChatConversationRequestError';
    this.status = status;
  }
}

export class DeleteChatConversationError extends Error {
  readonly status: number;

  constructor(status: number, options?: ErrorOptions) {
    super('Delete chat conversation failed', options);
    this.name = 'DeleteChatConversationError';
    this.status = status;
  }
}

export class StopChatStreamError extends Error {
  readonly status: number;

  constructor(status: number, options?: ErrorOptions) {
    super('Stop chat stream failed', options);
    this.name = 'StopChatStreamError';
    this.status = status;
  }
}

export const deleteChatConversation = async (
  conversationId: string,
): Promise<void> => {
  const response = await fetch(
    `/api/chat/${encodeURIComponent(conversationId)}`,
    {
      method: 'DELETE',
    },
  );

  if (!response.ok) {
    throw new DeleteChatConversationError(response.status);
  }
};

export const clearChatConversations = async (): Promise<void> => {
  const response = await fetch('/api/chat', { method: 'DELETE' });

  if (!response.ok) {
    throw new ChatConversationRequestError(response.status);
  }
};

const parseConversations = (value: unknown): ConversationRow[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (typeof item !== 'object' || item === null) {
      return [];
    }
    const candidate = item as Record<string, unknown>;
    const { id, model, title } = candidate;
    if (
      typeof id !== 'string' ||
      (model !== null && typeof model !== 'string') ||
      (title !== null && typeof title !== 'string')
    ) {
      return [];
    }
    return [{ id, model, title: title ?? 'New conversation' }];
  });
};

const MESSAGE_ROLES = new Set(['assistant', 'system', 'user']);
const TEXT_PART_TYPES = new Set(['reasoning', 'text']);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const isMessagePart = (value: unknown): value is Record<string, unknown> => {
  if (!isRecord(value) || typeof value['type'] !== 'string') {
    return false;
  }

  if (TEXT_PART_TYPES.has(value['type'])) {
    return typeof value['text'] === 'string';
  }

  return true;
};

const parseJsonOrNull = async (response: Response): Promise<unknown> => {
  try {
    return await response.json();
  } catch (error) {
    if (error instanceof SyntaxError) {
      return null;
    }
    throw error;
  }
};

const isUiMessage = (value: unknown): value is MyUIMessage => {
  if (!isRecord(value)) {
    return false;
  }
  const { id, parts, role } = value;
  return (
    typeof id === 'string' &&
    typeof role === 'string' &&
    MESSAGE_ROLES.has(role) &&
    Array.isArray(parts) &&
    parts.every(isMessagePart)
  );
};

export const listChatConversations = async (): Promise<ConversationRow[]> => {
  const response = await fetch('/api/chat', { method: 'GET' });

  if (!response.ok) {
    throw new ChatConversationRequestError(response.status);
  }

  return parseConversations(await parseJsonOrNull(response));
};

export type SaveChatConversationInput = {
  readonly expectedTitle?: string;
  readonly id: string;
  readonly model?: string;
  readonly title?: string;
};

export const saveChatConversation = async ({
  expectedTitle,
  id,
  model,
  title,
}: SaveChatConversationInput): Promise<void> => {
  const response = await fetch(`/api/chat/${encodeURIComponent(id)}`, {
    body: JSON.stringify({
      ...(expectedTitle !== undefined && { expectedTitle }),
      ...(model !== undefined && { model }),
      ...(title !== undefined && { title }),
    }),
    headers: { 'content-type': 'application/json' },
    method: 'PATCH',
  });

  if (!response.ok) {
    throw new ChatConversationRequestError(response.status);
  }
};

const parseConversationHistory = (
  value: unknown,
): ChatConversationHistory | null => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }
  const candidate = value as Record<string, unknown>;
  const { conversation, messages } = candidate;
  if (typeof conversation !== 'object' || conversation === null) {
    return null;
  }
  const conversationRecord = conversation as Record<string, unknown>;
  const { id, model, title } = conversationRecord;
  if (
    typeof id !== 'string' ||
    (model !== null && typeof model !== 'string') ||
    (title !== null && typeof title !== 'string') ||
    !Array.isArray(messages) ||
    !messages.every(isUiMessage)
  ) {
    return null;
  }
  return {
    conversation: { id, model, title: title ?? 'New conversation' },
    messages,
  };
};

export const loadChatConversationHistory = async (
  conversationId: string,
): Promise<ChatConversationHistory | null> => {
  const response = await fetch(
    `/api/chat/${encodeURIComponent(conversationId)}/history`,
    { method: 'GET' },
  );

  if (!response.ok) {
    throw new ChatConversationRequestError(response.status);
  }

  const body = await parseJsonOrNull(response);
  return parseConversationHistory(body);
};

export const stopChatStream = async (
  conversationId: string,
  options?: StopChatStreamOptions,
): Promise<void> => {
  const init: RequestInit =
    options === undefined
      ? { method: 'POST' }
      : {
          body: JSON.stringify(options),
          headers: { 'content-type': 'application/json' },
          method: 'POST',
        };
  const response = await fetch(
    `/api/chat/${encodeURIComponent(conversationId)}/stop`,
    init,
  );

  if (!response.ok) {
    throw new StopChatStreamError(response.status);
  }
};

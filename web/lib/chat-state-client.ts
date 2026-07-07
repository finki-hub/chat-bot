import 'server-only';

import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

/* eslint-disable camelcase -- Python chat state API uses snake_case fields. */

export type ChatStateClient = {
  readonly clearActiveStreamIfCurrent: (
    input: ClearActiveStreamInput,
  ) => Promise<void>;
  readonly loadConversation: (
    input: LoadConversationInput,
  ) => Promise<ChatStateConversationWithMessages>;
  readonly setActiveStream: (input: SetActiveStreamInput) => Promise<void>;
  readonly stopActiveStreamIfCurrent: (
    input: ClearActiveStreamInput,
  ) => Promise<void>;
  readonly upsertAssistantMessage: (
    input: UpsertAssistantMessageInput,
  ) => Promise<void>;
  readonly upsertConversation: (
    input: UpsertConversationInput,
  ) => Promise<void>;
  readonly upsertUserMessage: (input: UpsertUserMessageInput) => Promise<void>;
};

export type ChatStateConversation = {
  readonly active_response_id: null | string;
  readonly active_status: null | string;
  readonly active_stream_id: null | string;
  readonly created_at?: string;
  readonly id: string;
  readonly model?: null | string;
  readonly title?: null | string;
  readonly updated_at?: string;
  readonly user_id: string;
};

export type ChatStateConversationWithMessages = {
  readonly conversation: ChatStateConversation;
  readonly messages: readonly ChatStateJsonValue[];
};

export type ChatStateJsonValue =
  | boolean
  | null
  | number
  | readonly ChatStateJsonValue[]
  | string
  | { readonly [key: string]: ChatStateJsonValue };

export type ChatStateMetadata = Readonly<Record<string, ChatStateJsonValue>>;

type ClearActiveStreamInput = {
  readonly conversationId: string;
  readonly streamId: string;
  readonly userId: string;
};

type LoadConversationInput = {
  readonly conversationId: string;
  readonly userId: string;
};

type SetActiveStreamInput = {
  readonly activeResponseId: string;
  readonly activeStreamId: string;
  readonly conversationId: string;
  readonly userId: string;
};

type UpsertAssistantMessageInput = {
  readonly content: string;
  readonly conversationId: string;
  readonly metadata: ChatStateMetadata;
  readonly responseId: string;
  readonly userId: string;
};

type UpsertConversationInput = {
  readonly conversationId: string;
  readonly model?: string;
  readonly userId: string;
};

type UpsertUserMessageInput = {
  readonly content: string;
  readonly conversationId: string;
  readonly messageId: string;
  readonly userId: string;
};

export class ChatStateRequestError extends Error {
  readonly status: number;

  constructor(status: number, options?: ErrorOptions) {
    super('Chat state request failed', options);
    this.name = 'ChatStateRequestError';
    this.status = status;
  }
}

const stateUrl = (path: string): string => `${API_BASE_URL}/chat/state${path}`;

const sendStateRequest = async (
  path: string,
  init: Omit<RequestInit, 'headers'>,
): Promise<void> => {
  const response = await fetch(stateUrl(path), {
    ...init,
    headers: {
      'content-type': 'application/json',
      'x-api-key': CHAT_API_KEY,
    },
  });

  if (!response.ok) {
    throw new ChatStateRequestError(response.status);
  }
};

const readStateJson = async <T>(path: string): Promise<T> => {
  const response = await fetch(stateUrl(path), {
    headers: { 'x-api-key': CHAT_API_KEY },
    method: 'GET',
  });

  if (!response.ok) {
    throw new ChatStateRequestError(response.status);
  }

  return response.json() as Promise<T>;
};

export const createChatStateClient = (): ChatStateClient => ({
  clearActiveStreamIfCurrent: async ({ conversationId, streamId, userId }) => {
    await sendStateRequest(
      `/conversations/${conversationId}/active-stream/${streamId}?user_id=${encodeURIComponent(userId)}`,
      { method: 'DELETE' },
    );
  },
  loadConversation: async ({ conversationId, userId }) =>
    readStateJson<ChatStateConversationWithMessages>(
      `/conversations/${conversationId}?user_id=${encodeURIComponent(userId)}`,
    ),
  setActiveStream: async ({
    activeResponseId,
    activeStreamId,
    conversationId,
    userId,
  }) => {
    await sendStateRequest(`/conversations/${conversationId}/active-stream`, {
      body: JSON.stringify({
        active_response_id: activeResponseId,
        active_status: 'streaming',
        active_stream_id: activeStreamId,
        user_id: userId,
      }),
      method: 'PUT',
    });
  },
  stopActiveStreamIfCurrent: async ({ conversationId, streamId, userId }) => {
    await sendStateRequest(
      `/conversations/${conversationId}/active-stream/${streamId}/stop`,
      {
        body: JSON.stringify({ user_id: userId }),
        method: 'POST',
      },
    );
  },
  upsertAssistantMessage: async ({
    content,
    conversationId,
    metadata,
    responseId,
    userId,
  }) => {
    await sendStateRequest(
      `/conversations/${conversationId}/messages/assistant/${responseId}`,
      {
        body: JSON.stringify({
          content,
          id: crypto.randomUUID(),
          metadata,
          user_id: userId,
        }),
        method: 'PUT',
      },
    );
  },
  upsertConversation: async ({ conversationId, model, userId }) => {
    await sendStateRequest('/conversations', {
      body: JSON.stringify({
        id: conversationId,
        ...(model !== undefined && { model }),
        user_id: userId,
      }),
      method: 'POST',
    });
  },
  upsertUserMessage: async ({ content, conversationId, messageId, userId }) => {
    await sendStateRequest(`/conversations/${conversationId}/messages/user`, {
      body: JSON.stringify({
        content,
        id: messageId,
        metadata: {},
        user_id: userId,
      }),
      method: 'POST',
    });
  },
});

/* eslint-enable camelcase -- end Python chat state API snake_case fields. */

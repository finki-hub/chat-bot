import 'server-only';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
  ChatCredentialUpsert,
} from '@/lib/api-types';

import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

/* eslint-disable camelcase -- Python chat state API uses snake_case fields. */

export type ChatStateClient = {
  readonly clearActiveStreamIfCurrent: (
    input: ClearActiveStreamInput,
  ) => Promise<void>;
  readonly clearConversations: (input: UserScopedInput) => Promise<void>;
  readonly deleteConversation: (input: LoadConversationInput) => Promise<void>;
  readonly deleteCredential: (input: DeleteCredentialInput) => Promise<void>;
  readonly listConversations: (
    input: ListConversationsInput,
  ) => Promise<readonly ChatStateConversation[]>;
  readonly listCredentials: (
    input: UserScopedInput,
  ) => Promise<readonly ChatCredentialPublic[]>;
  readonly loadConversation: (
    input: LoadConversationInput,
  ) => Promise<ChatStateConversationWithMessages>;
  readonly replaceAssistantMessage: (
    input: ReplaceAssistantMessageInput,
  ) => Promise<void>;
  readonly setActiveStream: (input: SetActiveStreamInput) => Promise<void>;
  readonly stopActiveStreamIfCurrent: (
    input: ClearActiveStreamInput,
  ) => Promise<void>;
  readonly updateConversation: (
    input: UpdateConversationInput,
  ) => Promise<void>;
  readonly upsertAssistantMessage: (
    input: UpsertAssistantMessageInput,
  ) => Promise<void>;
  readonly upsertChatUser: (input: UpsertChatUserInput) => Promise<ChatUser>;
  readonly upsertConversation: (
    input: UpsertConversationInput,
  ) => Promise<void>;
  readonly upsertCredential: (
    input: UpsertCredentialInput,
  ) => Promise<ChatCredentialPublic>;
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

export type ChatUser = {
  readonly avatar_url: null | string;
  readonly created_at?: string;
  readonly email: null | string;
  readonly id: string;
  readonly name: null | string;
  readonly provider: ChatUserProvider;
  readonly provider_subject: string;
  readonly updated_at?: string;
};

export type ChatUserProvider = 'google' | 'microsoft-entra-id';

type ClearActiveStreamInput = {
  readonly conversationId: string;
  readonly streamId: string;
  readonly userId: string;
};

type DeleteCredentialInput = UserScopedInput & {
  readonly provider: ChatCredentialProvider;
};

type ListConversationsInput = {
  readonly limit?: number;
  readonly userId: string;
};

type LoadConversationInput = {
  readonly conversationId: string;
  readonly userId: string;
};

type ReplaceAssistantMessageInput = {
  readonly content: string;
  readonly conversationId: string;
  readonly messageId: string;
  readonly metadata: Record<string, ChatStateJsonValue>;
  readonly responseId: string;
  readonly retainedMessageIds: readonly string[];
  readonly userId: string;
};

type SetActiveStreamInput = {
  readonly activeResponseId: string;
  readonly activeStreamId: string;
  readonly conversationId: string;
  readonly userId: string;
};

type UpdateConversationInput = {
  readonly conversationId: string;
  readonly model?: string;
  readonly title?: string;
  readonly userId: string;
};

type UpsertAssistantMessageInput = {
  readonly content: string;
  readonly conversationId: string;
  readonly metadata: ChatStateMetadata;
  readonly responseId: string;
  readonly userId: string;
};

type UpsertChatUserInput = {
  readonly avatarUrl?: string;
  readonly email?: string;
  readonly name?: string;
  readonly provider: ChatUserProvider;
  readonly providerSubject: string;
};

type UpsertConversationInput = {
  readonly conversationId: string;
  readonly model?: string;
  readonly title?: string;
  readonly userId: string;
};

type UpsertCredentialInput = ChatCredentialUpsert & UserScopedInput;

type UpsertUserMessageInput = {
  readonly content: string;
  readonly conversationId: string;
  readonly messageId: string;
  readonly userId: string;
};

type UserScopedInput = {
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

const writeStateJson = async <T>(
  path: string,
  body: string,
  method: 'POST' | 'PUT',
): Promise<T> => {
  const response = await fetch(stateUrl(path), {
    body,
    headers: {
      'content-type': 'application/json',
      'x-api-key': CHAT_API_KEY,
    },
    method,
  });

  if (!response.ok) {
    throw new ChatStateRequestError(response.status);
  }

  return response.json() as Promise<T>;
};

const postStateJson = async <T>(path: string, body: string): Promise<T> =>
  writeStateJson(path, body, 'POST');

const putStateJson = async <T>(path: string, body: string): Promise<T> =>
  writeStateJson(path, body, 'PUT');

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
  clearConversations: async ({ userId }) => {
    await sendStateRequest(
      `/conversations?user_id=${encodeURIComponent(userId)}`,
      { method: 'DELETE' },
    );
  },
  deleteConversation: async ({ conversationId, userId }) => {
    await sendStateRequest(
      `/conversations/${conversationId}?user_id=${encodeURIComponent(userId)}`,
      { method: 'DELETE' },
    );
  },
  deleteCredential: async ({ provider, userId }) => {
    await sendStateRequest(
      `/users/${encodeURIComponent(userId)}/credentials/${provider}`,
      { method: 'DELETE' },
    );
  },
  listConversations: async ({ limit, userId }) =>
    readStateJson<readonly ChatStateConversation[]>(
      `/conversations?user_id=${encodeURIComponent(userId)}${
        limit === undefined ? '' : `&limit=${encodeURIComponent(String(limit))}`
      }`,
    ),
  listCredentials: async ({ userId }) =>
    readStateJson<readonly ChatCredentialPublic[]>(
      `/users/${encodeURIComponent(userId)}/credentials`,
    ),
  loadConversation: async ({ conversationId, userId }) =>
    readStateJson<ChatStateConversationWithMessages>(
      `/conversations/${conversationId}?user_id=${encodeURIComponent(userId)}`,
    ),
  replaceAssistantMessage: async ({
    content,
    conversationId,
    messageId,
    metadata,
    responseId,
    retainedMessageIds,
    userId,
  }) => {
    await sendStateRequest(
      `/conversations/${conversationId}/messages/assistant/${messageId}/replacement/${responseId}`,
      {
        body: JSON.stringify({
          content,
          metadata,
          retained_message_ids: retainedMessageIds,
          user_id: userId,
        }),
        method: 'PUT',
      },
    );
  },
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
  updateConversation: async ({ conversationId, model, title, userId }) => {
    await sendStateRequest(`/conversations/${conversationId}`, {
      body: JSON.stringify({
        ...(model !== undefined && { model }),
        ...(title !== undefined && { title }),
        user_id: userId,
      }),
      method: 'PATCH',
    });
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
  upsertChatUser: async ({
    avatarUrl,
    email,
    name,
    provider,
    providerSubject,
  }) =>
    postStateJson<ChatUser>(
      '/users',
      JSON.stringify({
        ...(avatarUrl !== undefined && { avatar_url: avatarUrl }),
        ...(email !== undefined && { email }),
        ...(name !== undefined && { name }),
        provider,
        provider_subject: providerSubject,
      }),
    ),
  upsertConversation: async ({ conversationId, model, title, userId }) => {
    await sendStateRequest('/conversations', {
      body: JSON.stringify({
        id: conversationId,
        ...(model !== undefined && { model }),
        ...(title !== undefined && { title }),
        user_id: userId,
      }),
      method: 'POST',
    });
  },
  upsertCredential: async ({ apiKey, baseUrl, provider, userId }) =>
    putStateJson<ChatCredentialPublic>(
      `/users/${encodeURIComponent(userId)}/credentials/${provider}`,
      JSON.stringify({
        api_key: apiKey,
        ...(baseUrl !== undefined && { base_url: baseUrl }),
        provider,
      }),
    ),
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

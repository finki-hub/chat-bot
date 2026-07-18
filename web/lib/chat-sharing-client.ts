import 'server-only';

import {
  type ChatStateJsonValue,
  ChatStateRequestError,
} from '@/lib/chat-state-client';
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

const SHARE_TOKEN_FIELD = 'share_token';
const USER_ID_FIELD = 'user_id';

export type ChatConversationShare = {
  readonly shareToken: string;
};

export type SharedChatConversation = {
  readonly conversation: {
    readonly title: null | string;
  };
  readonly messages: readonly ChatStateJsonValue[];
};

type ChatSharingClient = {
  readonly createConversationShare: (input: {
    readonly conversationId: string;
    readonly userId: string;
  }) => Promise<ChatConversationShare>;
  readonly getConversationShareStatus: (input: {
    readonly conversationId: string;
    readonly userId: string;
  }) => Promise<ChatConversationShare | null>;
  readonly loadSharedConversation: (input: {
    readonly shareToken: string;
  }) => Promise<SharedChatConversation>;
  readonly revokeConversationShare: (input: {
    readonly conversationId: string;
    readonly userId: string;
  }) => Promise<void>;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isJsonValue = (value: unknown): value is ChatStateJsonValue => {
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'number' ||
    typeof value === 'string'
  ) {
    return true;
  }
  if (Array.isArray(value)) {
    return value.every(isJsonValue);
  }
  if (isRecord(value)) {
    return Object.values(value).every(isJsonValue);
  }
  return false;
};

const invalidResponse = (): ChatStateRequestError =>
  new ChatStateRequestError(502);

const parseShare = (value: unknown): ChatConversationShare => {
  if (!isRecord(value) || typeof value[SHARE_TOKEN_FIELD] !== 'string') {
    throw invalidResponse();
  }
  return { shareToken: value[SHARE_TOKEN_FIELD] };
};

const parseSharedConversation = (value: unknown): SharedChatConversation => {
  if (!isRecord(value)) {
    throw invalidResponse();
  }
  const conversation = value['conversation'];
  const messages = value['messages'];
  if (!isRecord(conversation) || !Array.isArray(messages)) {
    throw invalidResponse();
  }
  const title = conversation['title'];
  if (
    (title !== null && title !== undefined && typeof title !== 'string') ||
    !messages.every(isJsonValue)
  ) {
    throw invalidResponse();
  }
  return {
    conversation: { title: title ?? null },
    messages,
  };
};

export const createChatSharingClient = (): ChatSharingClient => ({
  createConversationShare: async ({ conversationId, userId }) => {
    const response = await fetch(
      `${API_BASE_URL}/chat/state/conversations/${encodeURIComponent(conversationId)}/share`,
      {
        body: JSON.stringify({ [USER_ID_FIELD]: userId }),
        headers: {
          'content-type': 'application/json',
          'x-api-key': CHAT_API_KEY,
        },
        method: 'POST',
      },
    );
    if (!response.ok) {
      throw new ChatStateRequestError(response.status);
    }
    const body: unknown = await response.json();
    return parseShare(body);
  },
  getConversationShareStatus: async ({ conversationId, userId }) => {
    const userQuery = new URLSearchParams({ [USER_ID_FIELD]: userId });
    const response = await fetch(
      `${API_BASE_URL}/chat/state/conversations/${encodeURIComponent(conversationId)}/share?${userQuery.toString()}`,
      {
        headers: { 'x-api-key': CHAT_API_KEY },
        method: 'GET',
      },
    );
    if (response.status === 204) {
      return null;
    }
    if (!response.ok) {
      throw new ChatStateRequestError(response.status);
    }
    const body: unknown = await response.json();
    return parseShare(body);
  },
  loadSharedConversation: async ({ shareToken }) => {
    const response = await fetch(
      `${API_BASE_URL}/chat/state/shared/${encodeURIComponent(shareToken)}`,
      {
        cache: 'no-store',
        headers: { 'x-api-key': CHAT_API_KEY },
        method: 'GET',
      },
    );
    if (!response.ok) {
      throw new ChatStateRequestError(response.status);
    }
    const body: unknown = await response.json();
    return parseSharedConversation(body);
  },
  revokeConversationShare: async ({ conversationId, userId }) => {
    const response = await fetch(
      `${API_BASE_URL}/chat/state/conversations/${encodeURIComponent(conversationId)}/share`,
      {
        body: JSON.stringify({ [USER_ID_FIELD]: userId }),
        headers: {
          'content-type': 'application/json',
          'x-api-key': CHAT_API_KEY,
        },
        method: 'DELETE',
      },
    );
    if (!response.ok) {
      throw new ChatStateRequestError(response.status);
    }
  },
});

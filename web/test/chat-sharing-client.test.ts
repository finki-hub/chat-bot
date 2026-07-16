import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  API_BASE_URL,
  CONVERSATION_ID,
  SHARE_TOKEN,
  USER_ID,
} from './api-chat-route-support';

const RESPONSE_ID_FIELD = 'response_id';
const SHARE_TOKEN_FIELD = 'share_token';
const USER_ID_FIELD = 'user_id';

vi.mock('@/lib/env', () => ({
  API_BASE_URL,
  CHAT_API_KEY: 'test-key',
}));

describe('chat sharing client', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('creates a share through the API-key-protected owner endpoint', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json({ [SHARE_TOKEN_FIELD]: SHARE_TOKEN }));
    vi.stubGlobal('fetch', fetchMock);
    const { createChatSharingClient } =
      await import('@/lib/chat-sharing-client');

    const result = await createChatSharingClient().createConversationShare({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });

    expect(result).toStrictEqual({ shareToken: SHARE_TOKEN });
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/chat/state/conversations/${CONVERSATION_ID}/share`,
      {
        body: JSON.stringify({ [USER_ID_FIELD]: USER_ID }),
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'test-key',
        },
        method: 'POST',
      },
    );
  });

  it('reads whether an owned conversation is shared', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { createChatSharingClient } =
      await import('@/lib/chat-sharing-client');

    await expect(
      createChatSharingClient().getConversationShareStatus({
        conversationId: CONVERSATION_ID,
        userId: USER_ID,
      }),
    ).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/chat/state/conversations/${CONVERSATION_ID}/share?user_id=${encodeURIComponent(USER_ID)}`,
      {
        headers: { 'x-api-key': 'test-key' },
        method: 'GET',
      },
    );
  });

  it('reports an owned conversation without a share token', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(null, { status: 204 })),
    );
    const { createChatSharingClient } =
      await import('@/lib/chat-sharing-client');

    await expect(
      createChatSharingClient().getConversationShareStatus({
        conversationId: CONVERSATION_ID,
        userId: USER_ID,
      }),
    ).resolves.toBe(false);
  });

  it('revokes an owned conversation share', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    const { createChatSharingClient } =
      await import('@/lib/chat-sharing-client');

    await expect(
      createChatSharingClient().revokeConversationShare({
        conversationId: CONVERSATION_ID,
        userId: USER_ID,
      }),
    ).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/chat/state/conversations/${CONVERSATION_ID}/share`,
      {
        body: JSON.stringify({ [USER_ID_FIELD]: USER_ID }),
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'test-key',
        },
        method: 'DELETE',
      },
    );
  });

  it('loads shared history without caching it', async () => {
    const body = {
      conversation: { title: 'Shared enrollment chat' },
      messages: [
        {
          content: 'Question',
          id: 'message-1',
          metadata: {},
          parts: null,
          [RESPONSE_ID_FIELD]: null,
          role: 'user',
        },
      ],
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(body));
    vi.stubGlobal('fetch', fetchMock);
    const { createChatSharingClient } =
      await import('@/lib/chat-sharing-client');

    await expect(
      createChatSharingClient().loadSharedConversation({
        shareToken: SHARE_TOKEN,
      }),
    ).resolves.toStrictEqual(body);
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/chat/state/shared/${SHARE_TOKEN}`,
      {
        cache: 'no-store',
        headers: { 'x-api-key': 'test-key' },
        method: 'GET',
      },
    );
  });

  it('reports upstream failures with their status', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);
    const { createChatSharingClient } =
      await import('@/lib/chat-sharing-client');
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');

    await expect(
      createChatSharingClient().loadSharedConversation({
        shareToken: SHARE_TOKEN,
      }),
    ).rejects.toStrictEqual(new ChatStateRequestError(404));
  });
});

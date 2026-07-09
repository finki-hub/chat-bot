import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

const CHAT_USER_ID = '00000000-0000-4000-8000-000000000001';

describe('createChatStateClient', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('upserts an Auth.js user with the server API key', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json(
        Object.fromEntries([
          ['avatar_url', 'https://example.com/a.png'],
          ['email', 'student@example.com'],
          ['id', CHAT_USER_ID],
          ['name', 'Student'],
          ['provider', 'microsoft-entra-id'],
          ['provider_subject', 'microsoft-sub-1'],
        ]),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    const user = await createChatStateClient().upsertChatUser({
      avatarUrl: 'https://example.com/a.png',
      email: 'student@example.com',
      name: 'Student',
      provider: 'microsoft-entra-id',
      providerSubject: 'microsoft-sub-1',
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/chat/state/users');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');
    expect(JSON.parse(init.body as string)).toStrictEqual(
      Object.fromEntries([
        ['avatar_url', 'https://example.com/a.png'],
        ['email', 'student@example.com'],
        ['name', 'Student'],
        ['provider', 'microsoft-entra-id'],
        ['provider_subject', 'microsoft-sub-1'],
      ]),
    );
    expect(user.id).toBe(CHAT_USER_ID);
  });

  it('deletes a user-owned conversation with the server API key', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(null, {
        status: 200,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    await createChatStateClient().deleteConversation({
      conversationId: 'conv-delete',
      userId: CHAT_USER_ID,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(
      `https://api:8880/chat/state/conversations/conv-delete?user_id=${CHAT_USER_ID}`,
    );
    expect(init.method).toBe('DELETE');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');
  });

  it('loads a user-owned conversation with the server API key', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({
        conversation: Object.fromEntries([
          ['active_response_id', null],
          ['active_status', null],
          ['active_stream_id', null],
          ['id', 'conv-load'],
          ['user_id', CHAT_USER_ID],
        ]),
        messages: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    const loaded = await createChatStateClient().loadConversation({
      conversationId: 'conv-load',
      userId: CHAT_USER_ID,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(
      `https://api:8880/chat/state/conversations/conv-load?user_id=${CHAT_USER_ID}`,
    );
    expect(init.method).toBe('GET');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');
    expect(loaded.conversation.id).toBe('conv-load');
  });

  it('throws a typed request error when state writes fail', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(null, { status: 404 })),
    );

    const { ChatStateRequestError, createChatStateClient } =
      await import('@/lib/chat-state-client');

    await expect(
      createChatStateClient().setActiveStream({
        activeResponseId: 'response-missing',
        activeStreamId: 'stream-missing',
        conversationId: 'conv-missing',
        userId: CHAT_USER_ID,
      }),
    ).rejects.toMatchObject(new ChatStateRequestError(404));
  });
});

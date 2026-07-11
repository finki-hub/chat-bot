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

  it('lists user-owned conversations with the server API key', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json([
        Object.fromEntries([
          ['active_response_id', null],
          ['active_status', null],
          ['active_stream_id', null],
          ['id', 'conv-list'],
          ['model', 'model-a'],
          ['title', 'Listed'],
          ['user_id', CHAT_USER_ID],
        ]),
      ]),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    const conversations = await createChatStateClient().listConversations({
      userId: CHAT_USER_ID,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(
      `https://api:8880/chat/state/conversations?user_id=${CHAT_USER_ID}`,
    );
    expect(init.method).toBe('GET');
    expect(conversations[0]?.id).toBe('conv-list');
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

  it('clears user-owned conversations with the server API key', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null));
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    await createChatStateClient().clearConversations({
      userId: CHAT_USER_ID,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(
      `https://api:8880/chat/state/conversations?user_id=${CHAT_USER_ID}`,
    );
    expect(init.method).toBe('DELETE');
  });

  it('sends conversation titles on upsert and update requests', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null));
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    const client = createChatStateClient();
    await client.upsertConversation({
      conversationId: 'conv-title',
      model: 'model-a',
      title: 'Initial title',
      userId: CHAT_USER_ID,
    });
    await client.updateConversation({
      conversationId: 'conv-title',
      title: 'Renamed title',
      userId: CHAT_USER_ID,
    });

    const [, upsertInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    const [updateUrl, updateInit] = fetchMock.mock.calls[1] as [
      string,
      RequestInit,
    ];

    expect(JSON.parse(upsertInit.body as string)).toMatchObject({
      title: 'Initial title',
    });
    expect(updateUrl).toBe(
      'https://api:8880/chat/state/conversations/conv-title',
    );
    expect(updateInit.method).toBe('PATCH');
    expect(JSON.parse(updateInit.body as string)).toMatchObject({
      title: 'Renamed title',
    });
  });

  it('forwards BYOK credential writes without exposing the secret elsewhere', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json(
        Object.fromEntries([
          ['base_url', 'https://api.openai.com/v1'],
          ['has_api_key', true],
          ['provider', 'openai'],
          ['user_id', CHAT_USER_ID],
        ]),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    const credential = await createChatStateClient().upsertCredential({
      apiKey: 'user-secret-key',
      baseUrl: 'https://api.openai.com/v1',
      provider: 'openai',
      userId: CHAT_USER_ID,
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(
      `https://api:8880/chat/state/users/${CHAT_USER_ID}/credentials/openai`,
    );
    expect(init.method).toBe('PUT');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');
    expect(JSON.parse(init.body as string)).toStrictEqual(
      Object.fromEntries([
        ['api_key', 'user-secret-key'],
        ['base_url', 'https://api.openai.com/v1'],
        ['provider', 'openai'],
      ]),
    );
    expect(credential).toStrictEqual(
      Object.fromEntries([
        ['base_url', 'https://api.openai.com/v1'],
        ['has_api_key', true],
        ['provider', 'openai'],
        ['user_id', CHAT_USER_ID],
      ]),
    );
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

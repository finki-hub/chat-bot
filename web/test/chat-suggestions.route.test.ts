import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { getAuthenticatedChatUserIdMock } = vi.hoisted(() => ({
  getAuthenticatedChatUserIdMock: vi
    .fn<() => Promise<string>>()
    .mockResolvedValue('api-user-1'),
}));

const CONVERSATION_ID = '00000000-0000-4000-8000-000000000123';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));
vi.mock('@/lib/authenticated-chat-user', () => {
  class AuthenticationRequiredError extends Error {
    constructor() {
      super('Authentication required');
      this.name = 'AuthenticationRequiredError';
    }
  }

  return {
    AuthenticationRequiredError,
    getAuthenticatedChatUserId: getAuthenticatedChatUserIdMock,
  };
});

const jsonRequest = (body: unknown): Request =>
  new Request('https://localhost/api/chat/suggestions', {
    body: JSON.stringify(body),
    headers: {
      'content-type': 'application/json',
      'x-conversation-id': CONVERSATION_ID,
    },
    method: 'POST',
  });

const post = async (request: Request): Promise<Response> => {
  const { POST } = await import('@/app/api/chat/suggestions/route');

  return POST(request);
};

const fetchMock = vi.fn<typeof fetch>();

describe('/api/chat/suggestions', () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards the answer with the signed-in user and conversation', async () => {
    fetchMock.mockResolvedValue(
      Response.json({ suggestions: ['Прво?', '', 'Второ?', 'Трето?', 'Ч?'] }),
    );

    const response = await post(
      jsonRequest({
        answer: 'Уписот чини 200 денари.',
        providerModel: 'claude-sonnet-5-5',
        question: 'Колку чини уписот?',
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toStrictEqual({
      suggestions: ['Прво?', 'Второ?', 'Трето?'],
    });

    const [url, init] = fetchMock.mock.calls[0] ?? [];

    expect(url).toBe('https://api:8880/chat/suggestions');
    expect(JSON.parse(init?.body as string)).toStrictEqual({
      answer: 'Уписот чини 200 денари.',
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      provider_model: 'claude-sonnet-5-5',
      question: 'Колку чини уписот?',
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      user_id: 'api-user-1',
    });
    expect(new Headers(init?.headers).get('x-conversation-id')).toBe(
      CONVERSATION_ID,
    );
    expect(new Headers(init?.headers).get('x-api-key')).toBe('test-key');
  });

  it('rejects a payload without a question or answer', async () => {
    const response = await post(jsonRequest({ answer: 'само одговор' }));

    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects an oversized answer', async () => {
    const response = await post(
      jsonRequest({ answer: 'а'.repeat(8_001), question: 'Прашање?' }),
    );

    expect(response.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('reports an upstream failure as a bad gateway', async () => {
    fetchMock.mockResolvedValue(new Response('nope', { status: 500 }));

    const response = await post(
      jsonRequest({ answer: 'Одговор.', question: 'Прашање?' }),
    );

    expect(response.status).toBe(502);
  });
});

/* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'super-secret-key',
}));

const { authMock } = vi.hoisted(() => ({
  authMock: vi.fn<() => Promise<string>>(),
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
    getAuthenticatedChatUserId: authMock,
  };
});

const USER_ID = '00000000-0000-4000-8000-000000000101';

const importRoute = async (): Promise<{
  readonly GET: (request: Request) => Promise<Response>;
}> => {
  const { GET } = await import('@/app/api/chat/search/route');
  return { GET };
};

const searchResponse = {
  query: 'испити',
  results: [
    {
      conversation_id: '00000000-0000-4000-8000-000000000201',
      conversation_title: 'Испити',
      conversation_updated_at: '2026-11-02T12:00:00Z',
      message: {
        created_at: '2026-11-02T12:00:00Z',
        id: '00000000-0000-4000-8000-000000000301',
        role: 'assistant',
      },
      snippet: '\u{1}испити\u{2}',
    },
  ],
};

describe('GET /api/chat/search', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    authMock.mockResolvedValue(USER_ID);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards the authenticated user and maps the result to camelCase', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(searchResponse, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    const response = await GET(
      new Request(
        'https://localhost/api/chat/search?q=%D0%B8%D1%81%D0%BF%D0%B8%D1%82%D0%B8',
      ),
    );

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const upstream = new URL(url);

    expect(upstream.pathname).toBe('/chat/state/search');
    expect(upstream.searchParams.get('user_id')).toBe(USER_ID);
    expect(upstream.searchParams.get('q')).toBe('испити');
    expect(upstream.searchParams.get('limit')).toBe('20');
    expect(upstream.searchParams.get('offset')).toBe('0');
    expect(new Headers(init.headers).get('x-api-key')).toBe('super-secret-key');
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toStrictEqual({
      query: 'испити',
      results: [
        {
          conversationId: '00000000-0000-4000-8000-000000000201',
          conversationTitle: 'Испити',
          conversationUpdatedAt: '2026-11-02T12:00:00Z',
          message: {
            createdAt: '2026-11-02T12:00:00Z',
            id: '00000000-0000-4000-8000-000000000301',
            role: 'assistant',
          },
          snippet: '\u{1}испити\u{2}',
        },
      ],
    });
  });

  it('clamps explicit paging parameters', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(searchResponse, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    await GET(
      new Request(
        'https://localhost/api/chat/search?q=test&limit=999&offset=-5',
      ),
    );

    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    const upstream = new URL(url);

    expect(upstream.searchParams.get('limit')).toBe('50');
    expect(upstream.searchParams.get('offset')).toBe('0');
  });

  it('falls back when paging parameters are malformed', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(searchResponse, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    await GET(
      new Request('https://localhost/api/chat/search?q=test&limit=abc&offset='),
    );

    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    const upstream = new URL(url);

    expect(upstream.searchParams.get('limit')).toBe('20');
    expect(upstream.searchParams.get('offset')).toBe('0');
  });

  it('rejects a query shorter than two characters', async () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    const response = await GET(
      new Request('https://localhost/api/chat/search?q=a'),
    );

    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 401 for an unauthenticated browser', async () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    const { AuthenticationRequiredError } =
      await import('@/lib/authenticated-chat-user');
    authMock.mockRejectedValueOnce(new AuthenticationRequiredError());
    const { GET } = await importRoute();

    const response = await GET(
      new Request('https://localhost/api/chat/search?q=test'),
    );

    expect(response.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('preserves the upstream error status', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('{}', { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    const response = await GET(
      new Request('https://localhost/api/chat/search?q=test'),
    );

    expect(response.status).toBe(503);
  });
});

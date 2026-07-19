import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DELETE } from '@/app/api/feedback/route';

const { getAuthenticatedChatUserIdMock } = vi.hoisted(() => ({
  getAuthenticatedChatUserIdMock: vi
    .fn<() => Promise<string>>()
    .mockResolvedValue('api-user-1'),
}));

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'super-secret-key',
  env: {
    API_BASE_URL: 'https://api:8880',
    CHAT_API_KEY: 'super-secret-key',
  },
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

const deleteRequest = (body: unknown): Request =>
  new Request('https://localhost/api/feedback', {
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
    method: 'DELETE',
  });

describe('DELETE /api/feedback', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    getAuthenticatedChatUserIdMock.mockResolvedValue('api-user-1');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('injects the authenticated user and forwards only the response id', async () => {
    const ack = {
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      feedback_type: null,
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      response_id: 'r-123',
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(ack, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await DELETE(
      deleteRequest({
        feedbackType: 'like',
        responseId: 'r-123',
        userId: 'forged-user',
      }),
    );

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/chat/feedback');
    expect(init.method).toBe('DELETE');
    expect(new Headers(init.headers).get('x-api-key')).toBe('super-secret-key');
    /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
    expect(JSON.parse(init.body as string)).toStrictEqual({
      client: 'web',
      response_id: 'r-123',
      user_id: 'api-user-1',
    });
    /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toStrictEqual(ack);
  });

  it('returns 400 when responseId is missing', async () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);

    const response = await DELETE(deleteRequest({ feedbackType: 'like' }));

    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 401 when the browser is not authenticated', async () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    const { AuthenticationRequiredError } =
      await import('@/lib/authenticated-chat-user');
    getAuthenticatedChatUserIdMock.mockRejectedValueOnce(
      new AuthenticationRequiredError(),
    );

    const response = await DELETE(deleteRequest({ responseId: 'r-123' }));

    expect(response.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('preserves an upstream not-found response', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('{}', { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await DELETE(deleteRequest({ responseId: 'missing' }));

    expect(response.status).toBe(404);
  });
});

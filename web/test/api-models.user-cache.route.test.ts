import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GET } from '@/app/api/models/route';

/* eslint-disable camelcase -- fixtures mirror the API wire contract. */
const auth = vi.hoisted(() => ({
  getAuthenticatedChatUserId: vi.fn<() => Promise<string>>(),
}));

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

vi.mock('@/lib/authenticated-chat-user', () => auth);

const CACHE_HEADER = 'private, no-store';
const USER_A = 'user-a';
const USER_B = 'user-b';

const catalogFor = (remaining: number) => ({
  models: [
    {
      availability: 'sponsored',
      id: 'gpt-5.6-luna',
      name: 'GPT-5.6 Luna',
      provider: 'openai',
      sponsored_quota: {
        limit: 5,
        remaining,
        resets_at: '2099-01-01T12:00:00Z',
      },
    },
  ],
  source: 'live',
  version: 1,
});

describe('GET /api/models user-specific cache contract', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    auth.getAuthenticatedChatUserId.mockResolvedValue(USER_A);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does not reuse one user quota catalog for another authenticated subject', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json(catalogFor(5)))
      .mockResolvedValueOnce(Response.json(catalogFor(0)));
    vi.stubGlobal('fetch', fetchMock);

    const first = await GET();
    auth.getAuthenticatedChatUserId.mockResolvedValue(USER_B);
    const second = await GET();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      'https://api:8880/chat/models?user_id=user-a',
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      'https://api:8880/chat/models?user_id=user-b',
    );
    expect(first.headers.get('cache-control')).toBe(CACHE_HEADER);
    expect(second.headers.get('cache-control')).toBe(CACHE_HEADER);
    expect(first.headers.get('x-models-source')).toBe('live');
    expect(second.headers.get('x-models-source')).toBe('live');
    await expect(first.json()).resolves.toMatchObject({
      models: [{ sponsored_quota: { remaining: 5 } }],
    });
    await expect(second.json()).resolves.toMatchObject({
      models: [{ sponsored_quota: { remaining: 0 } }],
    });
  });

  it('omits cache headers for an upstream error catalog', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response('down', { status: 503 })),
    );

    const response = await GET();

    expect(response.headers.get('cache-control')).toBeNull();
    expect(response.headers.get('x-models-source')).toBe('error');
  });
});

/* eslint-enable camelcase -- end API wire fixtures. */

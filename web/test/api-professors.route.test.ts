import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'super-secret-key',
}));

const importRoute = async (): Promise<{
  readonly GET: () => Promise<Response>;
}> => {
  const { GET } = await import('@/app/api/professors/route');
  return { GET };
};

describe('GET /api/professors', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the public directory from the API with the server key', async () => {
    const professors = [
      {
        name: 'Александар Стојменски',
        position: 'Доцент',
        profile: 'https://finki.ukim.mk/kadar/aleksandar-stojmenski/',
      },
    ];
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(professors, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    const response = await GET();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/professors/');
    expect(new Headers(init.headers).get('x-api-key')).toBe('super-secret-key');
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toContain('max-age=300');
    await expect(response.json()).resolves.toStrictEqual(professors);
  });

  it('maps an upstream key failure to 502', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('{}', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    const response = await GET();

    expect(response.status).toBe(502);
  });

  it('returns 502 on a network failure', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);
    const { GET } = await importRoute();

    const response = await GET();

    expect(response.status).toBe(502);
  });
});

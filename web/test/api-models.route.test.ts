import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GET } from '@/app/api/models/route';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

const okJson = (body: unknown, init?: ResponseInit): Response =>
  Response.json(body, {
    headers: { 'content-type': 'application/json' },
    status: 200,
    ...init,
  });

describe('GET /api/models', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('proxies the upstream model list and returns a string[]', async () => {
    const models = ['claude-sonnet-4-6', 'gpt-5.4-mini', 'BAAI/bge-m3'];
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(okJson(models));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(fetchMock).toHaveBeenCalledOnce();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/chat/models');
    expect(init.method ?? 'GET').toBe('GET');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');

    expect(res.status).toBe(200);
    expect(res.headers.get('cache-control')).toBe(
      'public, max-age=300, stale-while-revalidate=600',
    );
    expect(res.headers.get('x-api-key')).toBeNull();
    await expect(res.json()).resolves.toStrictEqual(models);
  });

  it('returns [] with an error source header when upstream is non-2xx', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('nope', { status: 503 }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get('x-models-source')).toBe('error');
    await expect(res.json()).resolves.toStrictEqual([]);
  });

  it('returns [] with an error source header when fetch throws', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValue(new Error('network down'));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get('x-models-source')).toBe('error');
    await expect(res.json()).resolves.toStrictEqual([]);
  });

  it('returns [] when upstream JSON is not an array of strings', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(okJson({ models: 'oops' }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get('x-models-source')).toBe('error');
    await expect(res.json()).resolves.toStrictEqual([]);
  });
});

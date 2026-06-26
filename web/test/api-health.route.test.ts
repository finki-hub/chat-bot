import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GET } from '@/app/api/health/route';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

describe('GET /api/health', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reports ok when the upstream health check succeeds', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json({ status: 'ok' }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(fetchMock).toHaveBeenCalledOnce();

    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/health/health');
    expect(res.status).toBe(200);
    expect(res.headers.get('cache-control')).toBe('no-store');
    await expect(res.json()).resolves.toStrictEqual({ ok: true });
  });

  it('reports not ok when the upstream health check is unhealthy', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('unhealthy', { status: 503 }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(503);
    await expect(res.json()).resolves.toStrictEqual({ ok: false });
  });

  it('reports not ok when the upstream is unreachable', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValue(new Error('network down'));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(503);
    await expect(res.json()).resolves.toStrictEqual({ ok: false });
  });
});

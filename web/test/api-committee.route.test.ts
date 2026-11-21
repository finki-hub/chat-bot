/* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'super-secret-key',
}));

const committeeRequest = (
  body: unknown,
  headers: Record<string, string> = {},
): Request =>
  new Request('https://localhost/api/committee', {
    body: JSON.stringify(body),
    headers: {
      'content-type': 'application/json',
      'x-forwarded-for': '203.0.113.7',
      ...headers,
    },
    method: 'POST',
  });

const importRoute = async (): Promise<{
  readonly POST: (request: Request) => Promise<Response>;
}> => {
  const { POST } = await import('@/app/api/committee/route');
  return { POST };
};

describe('POST /api/committee', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards the proposal with the server key and no browser secret', async () => {
    const recommendation = {
      members: [],
      mentor: { name: 'Проф. д-р Тест' },
      mode: 'full',
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(recommendation, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    const response = await POST(
      committeeRequest({
        include_prior: false,
        title: 'Систем за препорака на ментори',
      }),
    );

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/recommendations/');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('x-api-key')).toBe('super-secret-key');
    expect(JSON.parse(init.body as string)).toStrictEqual({
      include_prior: false,
      title: 'Систем за препорака на ментори',
    });
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toStrictEqual(recommendation);
  });

  it('rejects invalid JSON without calling upstream', async () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    const response = await POST(
      new Request('https://localhost/api/committee', {
        body: '{not-json',
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      }),
    );

    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('maps an upstream key failure to 502 instead of leaking 401', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('{}', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    const response = await POST(committeeRequest({ title: 'Test' }));

    expect(response.status).toBe(502);
  });

  it('forces prior-off even when the client sends include_prior true', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json({ mode: 'full' }, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    await POST(committeeRequest({ include_prior: true, title: 'Test' }));

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as { include_prior: boolean };

    expect(body.include_prior).toBe(false);
  });

  it('preserves 422 validation errors from the recommender', async () => {
    const detail = { detail: 'mentor must be an active staff member' };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json(detail, { status: 422 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    const response = await POST(committeeRequest({ title: 'Test' }));

    expect(response.status).toBe(422);
    await expect(response.json()).resolves.toStrictEqual(detail);
  });

  it('rate limits repeated requests from the same client', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    for (let attempt = 0; attempt < 6; attempt += 1) {
      const response = await POST(
        committeeRequest({ title: `Test ${attempt}` }),
      );

      expect(response.status).toBe(200);
    }

    const limited = await POST(committeeRequest({ title: 'Test 7' }));

    expect(limited.status).toBe(429);
    expect(limited.headers.get('retry-after')).toBe('60');
    expect(fetchMock).toHaveBeenCalledTimes(6);
  });

  it('prefers the proxy-set X-Real-IP over a rotated forwarded prefix', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    for (let attempt = 0; attempt < 6; attempt += 1) {
      const response = await POST(
        committeeRequest(
          { title: `Test ${attempt}` },
          {
            'x-forwarded-for': `client-${attempt}`,
            'x-real-ip': '203.0.113.7',
          },
        ),
      );

      expect(response.status).toBe(200);
    }

    const limited = await POST(
      committeeRequest(
        { title: 'Test 7' },
        {
          'x-forwarded-for': 'client-99',
          'x-real-ip': '203.0.113.7',
        },
      ),
    );

    expect(limited.status).toBe(429);
  });

  it('uses the last forwarded hop when X-Real-IP is absent', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    for (let attempt = 0; attempt < 6; attempt += 1) {
      const response = await POST(
        committeeRequest(
          { title: `Test ${attempt}` },
          { 'x-forwarded-for': `client-${attempt}, 203.0.113.8` },
        ),
      );

      expect(response.status).toBe(200);
    }

    const limited = await POST(
      committeeRequest(
        { title: 'Test 7' },
        { 'x-forwarded-for': 'client-99, 203.0.113.8' },
      ),
    );

    expect(limited.status).toBe(429);
  });
});

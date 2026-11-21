/* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'super-secret-key',
}));

const CLIENT_IP = '203.0.113.7';
const RATE_LIMIT = 12;

const committeeRequest = (
  body: unknown,
  headers: Record<string, string> = {},
): Request =>
  new Request('https://localhost/api/committee', {
    body: JSON.stringify(body),
    headers: {
      'content-type': 'application/json',
      'x-forwarded-for': CLIENT_IP,
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

    for (let attempt = 0; attempt < RATE_LIMIT; attempt += 1) {
      const response = await POST(
        committeeRequest({ title: `Test ${attempt}` }),
      );

      expect(response.status).toBe(200);
    }

    const limited = await POST(committeeRequest({ title: 'Test 7' }));

    expect(limited.status).toBe(429);
    expect(limited.headers.get('retry-after')).toBe('60');
    expect(fetchMock).toHaveBeenCalledTimes(RATE_LIMIT);
  });

  it('prefers the proxy-set X-Real-IP over a rotated forwarded prefix', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    for (let attempt = 0; attempt < RATE_LIMIT; attempt += 1) {
      const response = await POST(
        committeeRequest(
          { title: `Test ${attempt}` },
          {
            'x-forwarded-for': `client-${attempt}`,
            'x-real-ip': CLIENT_IP,
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
          'x-real-ip': CLIENT_IP,
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

    for (let attempt = 0; attempt < RATE_LIMIT; attempt += 1) {
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

describe('POST /api/committee abuse limits', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('keys on the configured client IP header despite rotated proxy headers', async () => {
    vi.stubEnv('CLIENT_IP_HEADER', 'cf-connecting-ip');
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();
    const send = async (attempt: number, clientIp: string) =>
      POST(
        committeeRequest(
          { title: 'Test' },
          {
            'cf-connecting-ip': clientIp,
            'x-forwarded-for': `edge-${attempt}`,
            'x-real-ip': `spoofed-${attempt}`,
          },
        ),
      );

    for (let attempt = 0; attempt < RATE_LIMIT; attempt += 1) {
      const response = await send(attempt, CLIENT_IP);

      expect(response.status).toBe(200);
    }

    const limited = await send(RATE_LIMIT, CLIENT_IP);
    const otherClient = await send(RATE_LIMIT + 1, '203.0.113.8');

    expect(limited.status).toBe(429);
    expect(otherClient.status).toBe(200);
  });

  it('does not fall back to spoofable headers when the configured one is missing', async () => {
    vi.stubEnv('CLIENT_IP_HEADER', 'cf-connecting-ip');
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    for (let attempt = 0; attempt < RATE_LIMIT; attempt += 1) {
      await POST(
        committeeRequest(
          { title: 'Test' },
          { 'x-real-ip': `spoofed-${attempt}` },
        ),
      );
    }

    const limited = await POST(
      committeeRequest({ title: 'Test' }, { 'x-real-ip': 'spoofed-99' }),
    );

    expect(limited.status).toBe(429);
  });

  it('evicts the oldest client instead of rejecting new ones at capacity', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();

    for (let attempt = 0; attempt < 5_000; attempt += 1) {
      await POST(
        committeeRequest({ title: 'Test' }, { 'x-real-ip': `key-${attempt}` }),
      );
    }

    const newcomer = await POST(
      committeeRequest({ title: 'Test' }, { 'x-real-ip': '203.0.113.9' }),
    );

    expect(newcomer.status).toBe(200);
  });

  it('stops reading a chunked body once it exceeds the size cap', async () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    const { POST } = await importRoute();
    let pulls = 0;
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      cancel: () => {
        cancelled = true;
      },
      pull: (controller) => {
        pulls += 1;
        controller.enqueue(new Uint8Array(8_000));
      },
    });

    const response = await POST(
      new Request('https://localhost/api/committee', {
        body,
        duplex: 'half',
        method: 'POST',
      } as RequestInit),
    );

    expect(response.status).toBe(400);
    expect(cancelled).toBe(true);
    expect(pulls).toBeLessThan(10);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

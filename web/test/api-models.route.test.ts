import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/* eslint-disable camelcase -- fixtures mirror the Python API wire contract. */
import { GET } from '@/app/api/models/route';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

const USER_ID = '00000000-0000-4000-8000-000000000001';

vi.mock('@/lib/authenticated-chat-user', () => ({
  getAuthenticatedChatUserId: vi.fn<() => Promise<string>>(() =>
    Promise.resolve(USER_ID),
  ),
}));

const CACHE_CONTROL = 'cache-control';
const CACHE_HEADER = 'private, no-store';
const SOURCE_HEADER = 'x-models-source';
const LIVE = 'live';
const ERROR = 'error';
const STALE = 'stale';
const OPENAI = 'openai';
const ANTHROPIC = 'anthropic';
const GPT_MINI = 'gpt-5.4-mini';
const GPT_MINI_NAME = 'GPT-5.4 Mini';
const CLAUDE_5 = 'claude-sonnet-5';
const CLAUDE_5_NAME = 'Claude Sonnet 5';
const OLLAMA_MODEL = 'llama3.2:latest';

const okJson = (body: unknown, init?: ResponseInit): Response =>
  Response.json(body, {
    headers: { 'content-type': 'application/json' },
    status: 200,
    ...init,
  });

const gptMini = {
  id: GPT_MINI,
  name: GPT_MINI_NAME,
  provider: OPENAI,
};

const claudeSonnet = {
  id: CLAUDE_5,
  name: CLAUDE_5_NAME,
  provider: ANTHROPIC,
};

describe('GET /api/models', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards a typed catalog and keeps only the web-relevant fields', async () => {
    const upstream = {
      models: [
        {
          api_key: 'do-not-forward',
          availability: 'sponsored',
          base_url: 'https://sponsor.invalid',
          description: 'Balanced OpenAI model.',
          endpoint: 'https://sponsor.invalid/v1',
          execution: { reasoning: true },
          id: GPT_MINI,
          name: GPT_MINI_NAME,
          pricing: { input: 0.75, output: 4.5 },
          provider: OPENAI,
          sponsored_quota: {
            limit: 10,
            remaining: 7,
            resets_at: '2026-07-18T12:00:00Z',
          },
        },
        { ...claudeSonnet, loaded: false },
        {
          id: OLLAMA_MODEL,
          loaded: null,
          name: OLLAMA_MODEL,
          provider: 'ollama',
        },
      ],
      source: LIVE,
      version: 1,
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(okJson(upstream));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(fetchMock).toHaveBeenCalledOnce();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(
      `https://api:8880/chat/models?user_id=${encodeURIComponent(USER_ID)}`,
    );
    expect(init.method ?? 'GET').toBe('GET');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');

    expect(res.status).toBe(200);
    expect(res.headers.get(CACHE_CONTROL)).toBe(CACHE_HEADER);
    expect(res.headers.get('x-api-key')).toBeNull();
    expect(res.headers.get(SOURCE_HEADER)).toBe(LIVE);
    await expect(res.json()).resolves.toStrictEqual({
      models: [
        {
          availability: 'sponsored',
          description: 'Balanced OpenAI model.',
          id: GPT_MINI,
          name: GPT_MINI_NAME,
          provider: OPENAI,
          sponsored_quota: {
            limit: 10,
            remaining: 7,
            resets_at: '2026-07-18T12:00:00Z',
          },
        },
        { ...claudeSonnet, loaded: false },
        {
          id: OLLAMA_MODEL,
          loaded: null,
          name: OLLAMA_MODEL,
          provider: 'ollama',
        },
      ],
      source: LIVE,
      version: 1,
    });
  });

  it('normalizes a legacy string[] upstream into a typed catalog', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(okJson([CLAUDE_5, GPT_MINI]));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get(CACHE_CONTROL)).toBe(CACHE_HEADER);
    expect(res.headers.get(SOURCE_HEADER)).toBe(LIVE);
    await expect(res.json()).resolves.toStrictEqual({
      models: [
        {
          id: CLAUDE_5,
          name: CLAUDE_5_NAME,
          provider: ANTHROPIC,
        },
        { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI },
      ],
      source: LIVE,
      version: 1,
    });
  });

  it('propagates a stale source reported by the upstream', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        okJson({ models: [gptMini], source: STALE, version: 1 }),
      );

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.headers.get(SOURCE_HEADER)).toBe(STALE);
    await expect(res.json()).resolves.toMatchObject({ source: STALE });
  });

  it('returns an error catalog when upstream is non-2xx', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('nope', { status: 503 }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get(CACHE_CONTROL)).toBeNull();
    expect(res.headers.get(SOURCE_HEADER)).toBe(ERROR);
    await expect(res.json()).resolves.toStrictEqual({
      models: [],
      source: ERROR,
      version: 1,
    });
  });

  it('returns an error catalog when fetch throws', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValue(new Error('network down'));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get(SOURCE_HEADER)).toBe(ERROR);
    await expect(res.json()).resolves.toStrictEqual({
      models: [],
      source: ERROR,
      version: 1,
    });
  });

  it('returns an error catalog when upstream JSON is neither a catalog nor a string[]', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(okJson({ models: 'oops' }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get(SOURCE_HEADER)).toBe(ERROR);
    await expect(res.json()).resolves.toStrictEqual({
      models: [],
      source: ERROR,
      version: 1,
    });
  });
});

/* eslint-enable camelcase -- end wire-contract fixtures. */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  ChatTitleClientPayload,
  ChatTitleRequestBody,
  ChatTitleResponse,
} from '@/lib/api-types';

import { POST } from '@/app/api/chat/title/route';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

const jsonRequest = (body: unknown): Request =>
  new Request('https://localhost/api/chat/title', {
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
    method: 'POST',
  });

const okJson = (body: unknown): Response =>
  Response.json(body, {
    headers: { 'content-type': 'application/json' },
    status: 200,
  });

describe('POST /api/chat/title', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('proxies a title request to the chat API and returns the generated title', async () => {
    const payload: ChatTitleRequestBody = {
      messages: [{ content: 'Кога е испитот?', role: 'user' }],
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      query_transform_model: 'claude-sonnet-4-6',
    };
    const upstream: ChatTitleResponse = { title: 'Испитен рок' };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(okJson(upstream));

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(payload));

    expect(fetchMock).toHaveBeenCalledOnce();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/chat/title');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('content-type')).toBe(
      'application/json',
    );
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');
    expect(JSON.parse(init.body as string)).toStrictEqual({
      messages: [{ content: 'Кога е испитот?', role: 'user' }],
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      query_transform_model: 'claude-sonnet-4-6',
    });
    expect(res.headers.get('x-api-key')).toBeNull();
    await expect(res.json()).resolves.toStrictEqual(upstream);
  });

  it('also accepts the legacy camelCase model field', async () => {
    const payload: ChatTitleClientPayload = {
      messages: [{ content: 'Кога е испитот?', role: 'user' }],
      queryTransformModel: 'claude-sonnet-4-6',
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(okJson({ title: 'Испитен рок' }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(payload));
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(res.status).toBe(200);
    expect(JSON.parse(init.body as string)).toStrictEqual({
      messages: [{ content: 'Кога е испитот?', role: 'user' }],
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      query_transform_model: 'claude-sonnet-4-6',
    });
  });

  it('returns 400 when the payload has no messages', async () => {
    const fetchMock = vi.fn<typeof fetch>();

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest({ messages: [] }));

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 413 when the payload has too many messages', async () => {
    const fetchMock = vi.fn<typeof fetch>();

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({
        messages: [
          { content: '1', role: 'user' },
          { content: '2', role: 'assistant' },
          { content: '3', role: 'user' },
          { content: '4', role: 'assistant' },
          { content: '5', role: 'user' },
        ],
      }),
    );

    expect(res.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 413 when a message content is too long', async () => {
    const fetchMock = vi.fn<typeof fetch>();

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({
        messages: [{ content: 'x'.repeat(8_001), role: 'user' }],
      }),
    );

    expect(res.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 502 when the title service returns an invalid body', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(okJson({ title: '' }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({ messages: [{ content: 'Прашање?', role: 'user' }] }),
    );

    expect(res.status).toBe(502);
  });

  it('returns 502 when the title service returns malformed JSON', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response('not json', {
        headers: { 'content-type': 'application/json' },
        status: 200,
      }),
    );

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({ messages: [{ content: 'Прашање?', role: 'user' }] }),
    );

    expect(res.status).toBe(502);
  });
});

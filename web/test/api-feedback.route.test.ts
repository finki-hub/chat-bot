import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { FeedbackAck, FeedbackClientPayload } from '@/lib/api-types';

import { POST } from '@/app/api/feedback/route';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'super-secret-key',
  env: {
    API_BASE_URL: 'https://api:8880',
    CHAT_API_KEY: 'super-secret-key',
  },
}));

const jsonRequest = (body: unknown): Request =>
  new Request('https://localhost/api/feedback', {
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
    method: 'POST',
  });

const ack: FeedbackAck = {
  // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
  feedback_type: 'like',
  id: '00000000-0000-4000-8000-000000000000',
  // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
  response_id: 'r-123',
};

const okJson = (body: unknown): Response =>
  Response.json(body, {
    headers: { 'content-type': 'application/json' },
    status: 200,
  });

const validPayload: FeedbackClientPayload = {
  answerText: 'Испитот е на 1 јуни.',
  feedbackType: 'like',
  inferenceModel: 'claude-sonnet-4-6',
  questionText: 'Кога е испитот?',
  responseId: 'r-123',
  userId: 'anon-abc',
};

describe('POST /api/feedback', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('injects x-api-key and forwards a snake_case FeedbackSchema, returning the ack', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(okJson(ack));

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(validPayload));

    expect(fetchMock).toHaveBeenCalledOnce();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/chat/feedback');
    expect(init.method).toBe('POST');

    const headers = new Headers(init.headers);

    expect(headers.get('x-api-key')).toBe('super-secret-key');
    expect(headers.get('content-type')).toBe('application/json');

    /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
    expect(JSON.parse(init.body as string)).toStrictEqual({
      answer_text: 'Испитот е на 1 јуни.',
      client: 'web',
      feedback_type: 'like',
      inference_model: 'claude-sonnet-4-6',
      question_text: 'Кога е испитот?',
      response_id: 'r-123',
      user_id: 'anon-abc',
    });
    /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */

    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual(ack);
  });

  it('omits optional fields when not provided', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(okJson(ack));

    vi.stubGlobal('fetch', fetchMock);

    await POST(
      jsonRequest({
        feedbackType: 'dislike',
        responseId: 'r-123',
        userId: 'anon-abc',
      } satisfies FeedbackClientPayload),
    );

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
    expect(JSON.parse(init.body as string)).toStrictEqual({
      client: 'web',
      feedback_type: 'dislike',
      response_id: 'r-123',
      user_id: 'anon-abc',
    });
    /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */
  });

  it('never leaks the api key to the response or to the browser-facing body', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(okJson(ack));

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(validPayload));

    expect(res.headers.get('x-api-key')).toBeNull();
  });

  it('returns 400 when responseId is missing', async () => {
    const fetchMock = vi.fn<typeof fetch>();

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({ feedbackType: 'like', userId: 'anon-abc' }),
    );

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 400 when userId is empty', async () => {
    const fetchMock = vi.fn<typeof fetch>();

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({ feedbackType: 'like', responseId: 'r-123', userId: '' }),
    );

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 400 when feedbackType is not like/dislike', async () => {
    const fetchMock = vi.fn<typeof fetch>();

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({
        feedbackType: 'meh',
        responseId: 'r-123',
        userId: 'anon-abc',
      }),
    );

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 502 when upstream fails', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('boom', { status: 500 }));

    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(validPayload));

    expect(res.status).toBe(502);
    await expect(res.json()).resolves.toHaveProperty('error');
  });
});

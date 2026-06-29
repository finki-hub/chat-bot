import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const API_BASE_URL = 'https://api:8880';
const JSON_CONTENT_TYPE = 'application/json';
const MODEL = 'claude-sonnet-4-6';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

const sseBody = (...frames: string[]): ReadableStream<Uint8Array> => {
  const enc = new TextEncoder();

  return new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(enc.encode(frame));
      }

      controller.close();
    },
  });
};

describe('POST /api/chat', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('translates a python SSE answer into a UI message stream with metadata', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        sseBody(
          'event: sources\ndata: {"sources":[{"id":"c1","kind":"chunk","title":"Статут","section":"Член 12","snippet":"Правила."}]}\n\n',
          'event: token\ndata: {"text":"Здраво"}\n\n',
          'event: token\ndata: {"text":"!"}\n\n',
          'event: done\ndata: {}\n\n',
        ),
        {
          headers: {
            'content-type': 'text/event-stream',
            'X-Response-Id': 'resp-123',
          },
          status: 200,
        },
      ),
    );

    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('@/app/api/chat/route');
    const req = new Request('http://localhost/api/chat', {
      body: JSON.stringify({
        messages: [
          { id: 'u1', parts: [{ text: 'Здраво', type: 'text' }], role: 'user' },
        ],
        model: MODEL,
      }),
      headers: { 'content-type': JSON_CONTENT_TYPE },
      method: 'POST',
    });

    const res = await POST(req);

    expect(res.headers.get('content-type')).toContain('text/event-stream');

    expect(fetchMock).toHaveBeenCalledOnce();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe(`${API_BASE_URL}/chat/`);

    const sentBody = JSON.parse(init.body as string) as {
      inference_model?: string;
      interface?: string;
      messages: Array<{ content: string; role: string }>;
    };

    expect(sentBody.messages).toStrictEqual([
      { content: 'Здраво', role: 'user' },
    ]);
    expect(sentBody.interface).toBe('web');
    expect(sentBody.inference_model).toBe(MODEL);

    const out = await res.text();

    expect(out).toContain('Здраво');
    expect(out).toContain('resp-123');
    expect(out).toContain('Статут');
    expect(out).toContain('sources');
    expect(out).toContain('text-delta');
  });

  it('surfaces a pre-stream JSON error (503) as a data-error', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json(
        {
          detail: 'Failed to retrieve or re-rank context for the query.',
        },
        {
          headers: { 'content-type': JSON_CONTENT_TYPE },
          status: 503,
        },
      ),
    );

    vi.stubGlobal('fetch', fetchMock);

    const { POST } = await import('@/app/api/chat/route');
    const req = new Request('http://localhost/api/chat', {
      body: JSON.stringify({
        messages: [
          { id: 'u1', parts: [{ text: 'hi', type: 'text' }], role: 'user' },
        ],
      }),
      headers: { 'content-type': JSON_CONTENT_TYPE },
      method: 'POST',
    });

    const res = await POST(req);
    const out = await res.text();

    expect(out).toContain('data-error');
    expect(out).toContain('Failed to retrieve or re-rank context');
  });
});

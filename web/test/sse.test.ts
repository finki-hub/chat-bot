import { describe, expect, it } from 'vitest';

/* eslint-disable camelcase -- fixtures mirror the Python SSE wire contract. */
import { type ParsedEvent, parseProtocolV2 } from '@/lib/sse';

const DONE: ParsedEvent = { type: 'done' };
const DONE_FRAME = 'event: done\ndata: {}\n\n';
const SAFE_ERROR_MESSAGE = 'Request failed';

const token = (text: string): ParsedEvent => ({ text, type: 'token' });

const source = (chunks: string[]): AsyncIterable<string> => ({
  async *[Symbol.asyncIterator]() {
    for (const chunk of chunks) {
      yield await Promise.resolve(chunk);
    }
  },
});

const collect = async (...chunks: string[]): Promise<ParsedEvent[]> => {
  const out: ParsedEvent[] = [];
  const stream = parseProtocolV2(source(chunks));

  for await (const ev of stream) {
    out.push(ev);
  }

  return out;
};

describe('parseProtocolV2', () => {
  it('parses a plain token stream ending in done', async () => {
    const events = await collect(
      'event: token\ndata: {"text":"Здраво"}\n\n',
      'event: token\ndata: {"text":", свете"}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([token('Здраво'), token(', свете'), DONE]);
  });

  it('parses the tool path: status, reset, then answer tokens', async () => {
    const events = await collect(
      'event: status\ndata: {"state":"tool_call","label":"🔍 Пребарувам…","tool":"search_docs"}\n\n',
      'event: token\ndata: {"text":"некаков преамбула"}\n\n',
      'event: reset\ndata: {}\n\n',
      'event: token\ndata: {"text":"вистински одговор"}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([
      {
        label: '🔍 Пребарувам…',
        state: 'tool_call',
        tool: 'search_docs',
        type: 'status',
      },
      token('некаков преамбула'),
      { type: 'reset' },
      token('вистински одговор'),
      DONE,
    ]);
  });

  it('buffers a frame split across multiple chunks', async () => {
    const events = await collect(
      'event: token\nda',
      'ta: {"text":"спл',
      'ит"}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([token('сплит'), DONE]);
  });

  it('maps error frames to typed error events and clamps unknown codes', async () => {
    const events = await collect(
      'event: error\ndata: {"code":"interrupted","message":"provider secret: https://secret.invalid"}\n\n',
      'event: error\ndata: {"code":"credential_required","message":"provider detail: secret"}\n\n',
      'event: error\ndata: {"code":"weird","message":"provider detail: secret"}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([
      { code: 'interrupted', message: SAFE_ERROR_MESSAGE, type: 'error' },
      {
        code: 'credential_required',
        message: SAFE_ERROR_MESSAGE,
        type: 'error',
      },
      { code: 'agent_error', message: SAFE_ERROR_MESSAGE, type: 'error' },
      DONE,
    ]);
  });

  it('preserves reset metadata only for exhausted sponsored quota errors', async () => {
    const events = await collect(
      'event: error\ndata: {"code":"free_quota_exhausted","message":"quota provider secret","resets_at":"2026-07-18T12:00:00Z","endpoint":"https://secret.invalid"}\n\n',
      'event: error\ndata: {"code":"free_tier_unavailable","message":"offline provider detail","resets_at":"2026-07-18T12:00:00Z"}\n\n',
      'event: error\ndata: {"code":"sponsored_request_in_progress","message":"busy provider detail","resets_at":"2026-07-18T12:00:00Z"}\n\n',
      'event: error\ndata: {"code":"unknown_secret_error","message":"provider secret","resets_at":"2026-07-18T12:00:00Z"}\n\n',
    );

    expect(events).toStrictEqual([
      {
        code: 'free_quota_exhausted',
        message: SAFE_ERROR_MESSAGE,
        resets_at: '2026-07-18T12:00:00Z',
        type: 'error',
      },
      {
        code: 'free_tier_unavailable',
        message: SAFE_ERROR_MESSAGE,
        type: 'error',
      },
      {
        code: 'sponsored_request_in_progress',
        message: SAFE_ERROR_MESSAGE,
        type: 'error',
      },
      { code: 'agent_error', message: SAFE_ERROR_MESSAGE, type: 'error' },
    ]);
  });

  it('treats a bare data line as a token and un-escapes newlines', async () => {
    const events = await collect('data: прв ред\\nвтор ред\n\n');

    expect(events).toStrictEqual([token('прв ред\nвтор ред')]);
  });

  it('ignores unknown named events', async () => {
    const events = await collect('event: ping\ndata: {}\n\n', DONE_FRAME);

    expect(events).toStrictEqual([DONE]);
  });

  it('parses a thinking event into a thinking part', async () => {
    const events = await collect(
      'event: thinking\ndata: {"text":"размислувам"}\n\n',
      'event: token\ndata: {"text":"одговор"}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([
      { text: 'размислувам', type: 'thinking' },
      token('одговор'),
      DONE,
    ]);
  });

  it('maps meta frames to camelCase diagnostics, including a timing frame after done', async () => {
    // Mirrors the real wire order: tokens meta before done, timing meta after done.
    const events = await collect(
      'event: meta\ndata: {"tokens":{"input":12,"output":34,"total":46},"cost":{"input_usd":0.001,"output_usd":0.002,"total_usd":0.003}}\n\n',
      DONE_FRAME,
      'event: meta\ndata: {"timing":{"ttft_ms":120.5,"total_ms":980.2,"thinking_ms":340,"candidate_count":8,"top_distance":0.1234,"spans":{"retrieval.embed":42.1}}}\n\n',
    );

    expect(events).toStrictEqual([
      {
        diagnostics: {
          cost: { inputUsd: 0.001, outputUsd: 0.002, totalUsd: 0.003 },
          tokens: { input: 12, output: 34, total: 46 },
        },
        type: 'meta',
      },
      DONE,
      {
        diagnostics: {
          candidateCount: 8,
          serverTotalMs: 980.2,
          serverTtftMs: 120.5,
          spans: { 'retrieval.embed': 42.1 },
          thinkingMs: 340,
          topDistance: 0.1234,
        },
        type: 'meta',
      },
    ]);
  });

  it('maps source frames to typed retrieved sources and drops malformed entries', async () => {
    const events = await collect(
      'event: sources\ndata: {"sources":[{"id":"q1","kind":"faq","title":"Упис","links":[{"label":"iKnow","url":"https://iknow.ukim.mk/"}],"snippet":"Упис преку iKnow."},{"id":"bad","kind":"unknown","title":"bad"},{"id":"c1","kind":"chunk","title":"Статут","section":"Член 12","chunk_index":4,"snippet":"Правила."}]}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([
      {
        sources: [
          {
            id: 'q1',
            kind: 'faq',
            links: [{ label: 'iKnow', url: 'https://iknow.ukim.mk/' }],
            snippet: 'Упис преку iKnow.',
            title: 'Упис',
          },
          {
            chunkIndex: 4,
            id: 'c1',
            kind: 'chunk',
            section: 'Член 12',
            snippet: 'Правила.',
            title: 'Статут',
          },
        ],
        type: 'sources',
      },
      DONE,
    ]);
  });
});

/* eslint-enable camelcase -- end wire-contract fixtures. */

import { describe, expect, it } from 'vitest';

import { type ParsedEvent, parseProtocolV2 } from '@/lib/sse';

const DONE: ParsedEvent = { type: 'done' };
const DONE_FRAME = 'event: done\ndata: {}\n\n';

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
      'event: error\ndata: {"code":"interrupted","message":"одговорот е прекинат"}\n\n',
      'event: error\ndata: {"code":"weird","message":"boom"}\n\n',
      DONE_FRAME,
    );

    expect(events).toStrictEqual([
      { code: 'interrupted', message: 'одговорот е прекинат', type: 'error' },
      { code: 'agent_error', message: 'boom', type: 'error' },
      DONE,
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
      'event: meta\ndata: {"tokens":{"input":12,"output":34,"total":46}}\n\n',
      DONE_FRAME,
      'event: meta\ndata: {"timing":{"ttft_ms":120.5,"total_ms":980.2,"candidate_count":8,"top_distance":0.1234,"spans":{"retrieval.embed":42.1}}}\n\n',
    );

    expect(events).toStrictEqual([
      {
        diagnostics: { tokens: { input: 12, output: 34, total: 46 } },
        type: 'meta',
      },
      DONE,
      {
        diagnostics: {
          candidateCount: 8,
          serverTotalMs: 980.2,
          serverTtftMs: 120.5,
          spans: { 'retrieval.embed': 42.1 },
          topDistance: 0.1234,
        },
        type: 'meta',
      },
    ]);
  });
});

// Builds an AI SDK v5 "UI message stream" SSE body for Playwright route mocking.
// This is the BFF->browser wire format that useChat parses: each chunk is a JSON
// object emitted as `data: <json>\n\n`. We model exactly the chunks the real
// /api/chat translator (lib/chat-translate.ts) writes for a tool run.
import { createServer, type Server } from 'node:http';
import { type AddressInfo } from 'node:net';

export type UiChunk =
  | {
      data: { code: string; message: string };
      transient: true;
      type: 'data-error';
    }
  | {
      data: { label: string; tool?: string };
      transient: true;
      type: 'data-status';
    }
  | { delta: string; id: string; type: 'text-delta' }
  | { id: string; type: 'text-end' }
  | { id: string; type: 'text-start' }
  | {
      messageMetadata?: { inferenceModel?: string; responseId?: string };
      type: 'start';
    }
  | { type: 'finish' };

// Serialize chunks into an AI SDK UI-message-stream SSE body.
export const aiSdkStream = (
  chunks: UiChunk[],
): { body: string; contentType: string } => {
  const body = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');
  return { body: `${body}data: [DONE]\n\n`, contentType: 'text/event-stream' };
};

// Canonical "tool run" sequence, mirroring the real BFF translator
// (lib/chat-translate.ts) order verified by test/chat-translate.test.ts:
//   start(metadata) -> status chip -> preamble text part -> reset (new text part)
//   -> answer tokens -> finish.
// The status arrives BEFORE any text, so the client renders the chip while there
// is no text yet; once the preamble (then the answer) streams the chip hides and
// render-last drops the preamble, leaving only the answer on screen.
export const toolRunChunks = (opts: {
  answer: string;
  inferenceModel: string;
  preamble: string;
  responseId: string;
  statusLabel: string;
  tool: string;
}): UiChunk[] => {
  const preambleId = 'txt-preamble';
  const answerId = 'txt-answer';
  return [
    {
      messageMetadata: {
        inferenceModel: opts.inferenceModel,
        responseId: opts.responseId,
      },
      type: 'start',
    },
    // tool starts -> transient status chip (no text yet, so the chip shows):
    {
      data: { label: opts.statusLabel, tool: opts.tool },
      transient: true,
      type: 'data-status',
    },
    // preamble the model narrated before the tool resolved (dropped after reset):
    { id: preambleId, type: 'text-start' },
    { delta: opts.preamble, id: preambleId, type: 'text-delta' },
    { id: preambleId, type: 'text-end' },
    // reset -> a brand-new answer text part (render-last keeps only this one):
    { id: answerId, type: 'text-start' },
    { delta: opts.answer, id: answerId, type: 'text-delta' },
    { id: answerId, type: 'text-end' },
    { type: 'finish' },
  ];
};

// route.fulfill delivers its body atomically, so a single-shot mock lands the
// status chunk and the answer in one network read and React never paints the
// transient "searching…" chip on its own. To reproduce production timing (the
// BFF flushes chunks as the Python API emits them) we serve the SSE from a tiny
// local server that writes the head (start -> status), flushes, waits `gapMs`,
// then writes the tail (preamble -> reset -> answer -> finish -> [DONE]). The
// spec redirects the browser's same-origin POST /api/chat here, so the chip
// gets a real frame before the answer arrives.
const serialize = (chunks: UiChunk[]): string =>
  chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');

const closeServer = (server: Server): Promise<void> =>
  new Promise((resolve) => {
    server.close(() => {
      resolve();
    });
  });

export const startChatStreamServer = (opts: {
  gapMs: number;
  head: UiChunk[];
  tail: UiChunk[];
}): Promise<{ close: () => Promise<void>; url: string }> => {
  const handle: Parameters<typeof createServer>[1] = (req, res) => {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'access-control-allow-headers': '*',
        'access-control-allow-methods': '*',
        'access-control-allow-origin': '*',
      });
      res.end();
      return;
    }

    res.writeHead(200, {
      'access-control-allow-origin': '*',
      'cache-control': 'no-cache, no-transform',
      'content-type': 'text/event-stream',
      'x-vercel-ai-ui-message-stream': 'v1',
    });
    res.write(serialize(opts.head));
    setTimeout(() => {
      res.write(`${serialize(opts.tail)}data: [DONE]\n\n`);
      res.end();
    }, opts.gapMs);
  };

  return new Promise((resolve) => {
    const server = createServer(handle);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        close: () => closeServer(server),
        url: `http://127.0.0.1:${port}/chat`,
      });
    });
  });
};

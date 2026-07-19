import { createServer, type Server } from 'node:http';
import { type AddressInfo } from 'node:net';

export type UiChunk =
  | {
      data: Record<string, never>;
      transient: true;
      type: 'data-reset';
    }
  | {
      data: { code: string; message: string };
      transient: true;
      type: 'data-error';
    }
  | {
      data: { label: string; stage?: string; tool?: string };
      transient: true;
      type: 'data-status';
    }
  | { delta: string; id: string; type: 'reasoning-delta' }
  | { delta: string; id: string; type: 'text-delta' }
  | { id: string; type: 'reasoning-end' }
  | { id: string; type: 'reasoning-start' }
  | { id: string; type: 'text-end' }
  | { id: string; type: 'text-start' }
  | {
      messageMetadata: {
        diagnostics: {
          cost?: { inputUsd: number; outputUsd: number; totalUsd: number };
          serverTotalMs: number;
          serverTtftMs: number;
          spans?: Record<string, number>;
          tokens: { input: number; output: number; total: number };
        };
        inferenceModel?: string;
        responseId?: string;
      };
      type: 'message-metadata';
    }
  | {
      messageMetadata?: { inferenceModel?: string; responseId?: string };
      type: 'start';
    }
  | { type: 'finish' };

export const aiSdkStream = (
  chunks: UiChunk[],
): { body: string; contentType: string } => {
  const body = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');
  return { body: `${body}data: [DONE]\n\n`, contentType: 'text/event-stream' };
};

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
    {
      data: { label: opts.statusLabel, tool: opts.tool },
      transient: true,
      type: 'data-status',
    },
    { id: preambleId, type: 'text-start' },
    { delta: opts.preamble, id: preambleId, type: 'text-delta' },
    { id: preambleId, type: 'text-end' },
    { data: {}, transient: true, type: 'data-reset' },
    { id: answerId, type: 'text-start' },
    { delta: opts.answer, id: answerId, type: 'text-delta' },
    { id: answerId, type: 'text-end' },
    { type: 'finish' },
  ];
};

export const stagedRunChunks = (opts: {
  answer: string;
  inferenceModel: string;
  responseId: string;
  statusLabel: string;
  tool: string;
}): UiChunk[] => {
  const answerId = 'txt-answer';
  const stage = (s: string, tool?: string): UiChunk => ({
    data: { label: opts.statusLabel, stage: s, ...(tool && { tool }) },
    transient: true,
    type: 'data-status',
  });
  return [
    {
      messageMetadata: {
        inferenceModel: opts.inferenceModel,
        responseId: opts.responseId,
      },
      type: 'start',
    },
    stage('contextualize'),
    stage('retrieve', opts.tool),
    stage('rerank'),
    stage('context'),
    { data: {}, transient: true, type: 'data-reset' },
    { id: answerId, type: 'text-start' },
    { delta: opts.answer, id: answerId, type: 'text-delta' },
    { id: answerId, type: 'text-end' },
    { type: 'finish' },
  ];
};

const serialize = (chunks: UiChunk[]): string =>
  chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');

const closeServer = (server: Server): Promise<void> =>
  new Promise((resolve) => {
    server.close(() => {
      resolve();
    });
    server.closeAllConnections();
  });

export const startChatStreamServer = (opts: {
  gapMs: number;
  head: UiChunk[];
  tail: UiChunk[];
}): Promise<{ close: () => Promise<void>; url: string }> => {
  const timers = new Set<ReturnType<typeof setTimeout>>();
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
    const timer = setTimeout(() => {
      timers.delete(timer);
      res.write(`${serialize(opts.tail)}data: [DONE]\n\n`);
      res.end();
    }, opts.gapMs);
    timers.add(timer);
  };

  return new Promise((resolve) => {
    const server = createServer(handle);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        close: async () => {
          for (const timer of timers) {
            clearTimeout(timer);
          }
          timers.clear();
          await closeServer(server);
        },
        url: `http://127.0.0.1:${port}/chat`,
      });
    });
  });
};

import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';
import { type AddressInfo } from 'node:net';

import { type UiChunk } from './helpers/sse';

const RESPONSE_ID = '99999999-8888-7777-6666-555555555555';
const INFERENCE_MODEL = 'claude-sonnet-4-6';
const REASON_HEAD = 'Размислувам чекор еден…';
const REASON_TAIL = ' и чекор два.';
const ANSWER = 'Конечниот одговор е дека ФИНКИ е во Скопје.';

const frame = (chunk: UiChunk): string => `data: ${JSON.stringify(chunk)}\n\n`;

const closeServer = (server: ReturnType<typeof createServer>): Promise<void> =>
  new Promise((resolve) => {
    server.close(() => {
      resolve();
    });
  });

// A streaming server that flushes each chunk on its own schedule, so reasoning deltas
// genuinely arrive over time (the shared head/tail helper batches them).
const startTimedStream = (
  events: Array<{ atMs: number; chunk?: UiChunk; done?: boolean }>,
): Promise<{ close: () => Promise<void>; url: string }> =>
  new Promise((resolve) => {
    const server = createServer((req, res) => {
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
      res.flushHeaders();
      for (const event of events) {
        setTimeout(() => {
          if (event.done) {
            res.write('data: [DONE]\n\n');
            res.end();
          } else if (event.chunk) {
            res.write(frame(event.chunk));
          }
        }, event.atMs);
      }
    });
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        close: () => closeServer(server),
        url: `http://127.0.0.1:${port}/chat`,
      });
    });
  });

test.describe('reasoning streaming (mocked BFF)', () => {
  test('streams reasoning live in an auto-expanded panel before the answer, then collapses', async ({
    page,
  }) => {
    const chatServer = await startTimedStream([
      {
        atMs: 0,
        chunk: {
          messageMetadata: {
            inferenceModel: INFERENCE_MODEL,
            responseId: RESPONSE_ID,
          },
          type: 'start',
        },
      },
      { atMs: 0, chunk: { id: 'r', type: 'reasoning-start' } },
      {
        atMs: 0,
        chunk: { delta: REASON_HEAD, id: 'r', type: 'reasoning-delta' },
      },
      // reasoning keeps streaming for ~1.2s before any answer text exists
      {
        atMs: 700,
        chunk: { delta: REASON_TAIL, id: 'r', type: 'reasoning-delta' },
      },
      { atMs: 1_200, chunk: { id: 'r', type: 'reasoning-end' } },
      { atMs: 1_400, chunk: { id: 'a', type: 'text-start' } },
      { atMs: 1_400, chunk: { delta: ANSWER, id: 'a', type: 'text-delta' } },
      { atMs: 1_400, chunk: { id: 'a', type: 'text-end' } },
      { atMs: 1_400, chunk: { type: 'finish' } },
      { atMs: 1_500, done: true },
    ]);

    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: '{}',
        contentType: 'application/json',
        status: 200,
      });
    });
    await page.route('**/api/models', async (route) => {
      await route.fulfill({
        body: JSON.stringify([INFERENCE_MODEL, 'gpt-5.4-mini']),
        contentType: 'application/json',
        status: 200,
      });
    });
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        headers: { location: chatServer.url },
        status: 307,
      });
    });

    await page.goto('/');
    await page.getByTestId('composer-input').fill('Каде е ФИНКИ?');
    await page.getByTestId('composer-input').press('Enter');

    // Mid-stream: the panel is auto-expanded and the partial reasoning is visible
    // BEFORE the answer text exists.
    await expect(page.getByTestId('reasoning-panel')).toContainText(
      'чекор еден',
    );
    await expect(page.getByTestId('answer-text')).toHaveCount(0);

    // After the stream completes: the answer renders and the panel auto-collapses
    // (the toggle stays, so the reasoning is still reachable).
    await expect(page.getByTestId('answer-text')).toContainText(
      'ФИНКИ е во Скопје',
    );
    await expect(page.getByTestId('reasoning')).toBeVisible();
    await expect(page.getByTestId('reasoning-panel')).toHaveCount(0);

    // Re-expanding shows the full reasoning trace.
    await page.getByTestId('reasoning').getByRole('button').click();
    await expect(page.getByTestId('reasoning-panel')).toContainText(
      'чекор еден… и чекор два.',
    );

    await chatServer.close();
  });
});

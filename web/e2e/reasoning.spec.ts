import { expect, test } from '@playwright/test';
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from 'node:http';
import { type AddressInfo } from 'node:net';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { type UiChunk } from './helpers/sse';

const RESPONSE_ID = '99999999-8888-7777-6666-555555555555';
const INFERENCE_MODEL = 'claude-sonnet-5';
const REASON_HEAD = 'Размислувам чекор еден…';
const REASON_TAIL = ' и чекор два.';
const ANSWER = 'Конечниот одговор е дека ФИНКИ е во Скопје.';

type TimedEvent = { atMs: number; chunk?: UiChunk; done?: boolean };

const frame = (chunk: UiChunk): string => `data: ${JSON.stringify(chunk)}\n\n`;

const closeServer = (server: ReturnType<typeof createServer>): Promise<void> =>
  new Promise((resolve) => {
    server.close(() => {
      resolve();
    });
  });

// Skip a late timer once the client navigated away: writing to an ended socket throws.
const writeEvent = (res: ServerResponse, event: TimedEvent): void => {
  if (res.writableEnded || res.destroyed) {
    return;
  }
  if (event.done) {
    res.write('data: [DONE]\n\n');
    res.end();
  } else if (event.chunk) {
    res.write(frame(event.chunk));
  }
};

const handleStream =
  (events: TimedEvent[]) =>
  (req: IncomingMessage, res: ServerResponse): void => {
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
    const timers = events.map((event) =>
      setTimeout(() => {
        writeEvent(res, event);
      }, event.atMs),
    );
    res.on('close', () => {
      for (const timer of timers) {
        clearTimeout(timer);
      }
    });
  };

// Flushes each chunk on its own timer so reasoning deltas arrive over time, not batched.
const startTimedStream = (
  events: TimedEvent[],
): Promise<{ close: () => Promise<void>; url: string }> =>
  new Promise((resolve) => {
    const server = createServer(handleStream(events));
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

    try {
      await page.route('**/api/health', async (route) => {
        await route.fulfill({
          body: '{}',
          contentType: 'application/json',
          status: 200,
        });
      });
      await mockModels(page);
      await installMockChatState(page, { streamUrl: chatServer.url });

      await page.goto('/');
      await page.getByTestId('composer-input').fill('Каде е ФИНКИ?');
      await page.getByTestId('composer-input').press('Enter');

      // Reasoning is visible mid-stream, before any answer text exists.
      await expect(page.getByTestId('reasoning-panel')).toContainText(
        'чекор еден',
      );
      await expect(page.getByTestId('answer-text')).toHaveCount(0);

      // Once the answer arrives the panel auto-collapses, but the toggle remains.
      await expect(page.getByTestId('answer-text')).toContainText(
        'ФИНКИ е во Скопје',
      );
      await expect(page.getByTestId('reasoning')).toBeVisible();
      await expect(page.getByTestId('reasoning-panel')).toHaveCount(0);

      await page.getByTestId('reasoning').getByRole('button').click();
      await expect(page.getByTestId('reasoning-panel')).toContainText(
        'чекор еден… и чекор два.',
      );
    } finally {
      await chatServer.close();
    }
  });

  test('keeps persisted reasoning and diagnostics visible after switching chats', async ({
    page,
  }) => {
    const firstId = '11111111-1111-4111-8111-111111111111';
    const secondId = '22222222-2222-4222-8222-222222222222';
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: '{}',
        contentType: 'application/json',
        status: 200,
      });
    });
    await mockModels(page);
    await installMockChatState(page, {
      conversations: [
        { id: firstId, model: INFERENCE_MODEL, title: 'Reasoning history' },
        { id: secondId, model: INFERENCE_MODEL, title: 'Other history' },
      ],
      histories: {
        [firstId]: {
          conversation: {
            id: firstId,
            model: INFERENCE_MODEL,
            title: 'Reasoning history',
          },
          messages: [
            {
              id: 'assistant-reasoning-history',
              metadata: {
                diagnostics: {
                  serverTotalMs: 1_700,
                  serverTtftMs: 200,
                  thinkingMs: 400,
                },
                inferenceModel: INFERENCE_MODEL,
                responseId: RESPONSE_ID,
                timing: { totalMs: 1_700, ttftMs: 200 },
              },
              parts: [
                { state: 'done', text: 'Stored reasoning', type: 'reasoning' },
                { state: 'done', text: 'Stored answer', type: 'text' },
              ],
              role: 'assistant',
            },
          ],
        },
        [secondId]: {
          conversation: {
            id: secondId,
            model: INFERENCE_MODEL,
            title: 'Other history',
          },
          messages: [
            {
              id: 'assistant-other-history',
              metadata: {},
              parts: [{ text: 'Other answer', type: 'text' }],
              role: 'assistant',
            },
          ],
        },
      },
      streamUrl: 'http://127.0.0.1:9/unused',
    });

    await page.goto('/');
    await page
      .getByRole('button', { exact: true, name: 'Reasoning history' })
      .click();
    await expect(page.getByTestId('reasoning')).toBeVisible();
    await expect(page.getByTestId('message-timing')).toBeVisible();

    await page
      .getByRole('button', { exact: true, name: 'Other history' })
      .click();
    await expect(page.getByText('Other answer', { exact: true })).toBeVisible();
    await page
      .getByRole('button', { exact: true, name: 'Reasoning history' })
      .click();

    await expect(
      page.getByText('Stored answer', { exact: true }),
    ).toBeVisible();
    await page.getByTestId('reasoning').getByRole('button').click();
    await expect(page.getByTestId('reasoning-panel')).toContainText(
      'Stored reasoning',
    );
    await page.getByTestId('message-timing').hover();
    await expect(page.getByText(RESPONSE_ID)).toBeVisible();
  });
});

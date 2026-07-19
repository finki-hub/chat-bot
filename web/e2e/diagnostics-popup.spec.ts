import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer, toolRunChunks } from './helpers/sse';

const RESPONSE_ID = '11111111-2222-3333-4444-555555555555';
const INFERENCE_MODEL = 'claude-sonnet-5';
const ANSWER = 'Резултатите од испитите се објавуваат на https://finki.ukim.mk';
const DIAGNOSTICS_LABEL = /Дијагностика/u;

test('keeps complete diagnostics reachable inside a small viewport', async ({
  page,
}) => {
  await page.setViewportSize({ height: 360, width: 320 });
  const chatChunks = toolRunChunks({
    answer: ANSWER,
    inferenceModel: INFERENCE_MODEL,
    preamble: 'Дозволете да проверам…',
    responseId: RESPONSE_ID,
    statusLabel: 'Пребарувам…',
    tool: 'search_documents',
  });
  const diagnosticsChunk = {
    messageMetadata: {
      diagnostics: {
        cost: { inputUsd: 0.00003, outputUsd: 0.00045, totalUsd: 0.00048 },
        serverTotalMs: 1_700,
        serverTtftMs: 200,
        spans: Object.fromEntries(
          Array.from({ length: 12 }, (_, index) => [`stage.${index + 1}`, 100]),
        ),
        tokens: { input: 10, output: 30, total: 40 },
      },
      inferenceModel: INFERENCE_MODEL,
      responseId: RESPONSE_ID,
    },
    type: 'message-metadata',
  } satisfies (typeof chatChunks)[number];
  const chunks = [
    ...chatChunks.slice(0, -1),
    diagnosticsChunk,
    ...chatChunks.slice(-1),
  ];
  const chatServer = await startChatStreamServer({
    gapMs: 0,
    head: chunks,
    tail: [],
  });

  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page);
  await installMockChatState(page, { streamUrl: chatServer.url });
  await page.goto('/');
  const input = page.getByTestId('composer-input');
  await input.fill('Кога се објавуваат резултатите?');
  await page.getByTestId('composer-submit').click();
  await expect(page.getByTestId('answer-text')).toContainText(
    'Резултатите од испитите се објавуваат',
  );

  await page.getByRole('button', { name: DIAGNOSTICS_LABEL }).focus();
  const diagnosticsPopup = page.locator('[data-slot="hover-card-content"]');
  await expect(diagnosticsPopup).toBeVisible();
  const layout = await diagnosticsPopup.evaluate((element) => {
    const bounds = element.getBoundingClientRect();

    return {
      bottom: bounds.bottom,
      clientHeight: element.clientHeight,
      left: bounds.left,
      right: bounds.right,
      scrollHeight: element.scrollHeight,
      top: bounds.top,
    };
  });

  expect(layout.left).toBeGreaterThanOrEqual(15);
  expect(layout.right).toBeLessThanOrEqual(305);
  expect(layout.top).toBeGreaterThanOrEqual(15);
  expect(layout.bottom).toBeLessThanOrEqual(345);
  expect(layout.scrollHeight).toBeGreaterThan(layout.clientHeight);

  await diagnosticsPopup.evaluate((element) => {
    element.scrollTo({ top: element.scrollHeight });
  });
  await expect
    .poll(() =>
      diagnosticsPopup.evaluate(
        (element) =>
          element.scrollTop + element.clientHeight >= element.scrollHeight - 1,
      ),
    )
    .toBe(true);

  await chatServer.close();
});

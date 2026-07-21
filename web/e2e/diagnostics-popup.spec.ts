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
    gapMs: 100,
    head: chunks.slice(0, -1),
    tail: chunks.slice(-1),
  });

  try {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: JSON.stringify({ ok: true }),
        contentType: 'application/json',
        status: 200,
      });
    });
    await mockModels(page);
    await page.route('**/api/chat/*/stream', async (route) => {
      await route.fulfill({ status: 204 });
    });
    await installMockChatState(page, { streamUrl: chatServer.url });
    await page.goto('/');
    const input = page.getByTestId('composer-input');
    const submit = page.getByTestId('composer-submit');
    await input.fill('Кога се објавуваат резултатите?');
    const resumeResponsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname.endsWith('/stream'),
    );
    await submit.click();
    const resumeResponse = await resumeResponsePromise;
    expect(resumeResponse.status()).toBe(204);
    await expect(page.getByTestId('answer-text')).toContainText(
      'Резултатите од испитите се објавуваат',
    );
    await expect(submit).toHaveAccessibleName('Испрати');
    await expect(
      page.getByText('Се случи неочекувана грешка. Обидете се повторно.', {
        exact: true,
      }),
    ).toHaveCount(0);
    await page.getByRole('button', { name: DIAGNOSTICS_LABEL }).focus();
    const diagnosticsPopup = page.locator('[data-slot="hover-card-content"]');
    const waitForPopupAnimation = async () => {
      await diagnosticsPopup.evaluate(async (element) => {
        await Promise.allSettled(
          element.getAnimations().map((animation) => animation.finished),
        );
      });
    };
    await expect(diagnosticsPopup).toBeVisible();
    await waitForPopupAnimation();
    await expect
      .poll(() =>
        diagnosticsPopup.evaluate(
          (element) => element.scrollHeight > element.clientHeight,
        ),
      )
      .toBe(true);

    await page.keyboard.press('PageDown');
    await expect
      .poll(() => diagnosticsPopup.evaluate((element) => element.scrollTop))
      .toBeGreaterThan(0);

    await page.keyboard.press('End');
    await expect
      .poll(() =>
        diagnosticsPopup.evaluate(
          (element) =>
            element.scrollTop + element.clientHeight >=
            element.scrollHeight - 1,
        ),
      )
      .toBe(true);

    await expect
      .poll(() =>
        diagnosticsPopup.evaluate((element) => {
          const bounds = element.getBoundingClientRect();

          return [
            ...(bounds.left < 15 ? [`left=${bounds.left}`] : []),
            ...(bounds.right > 305 ? [`right=${bounds.right}`] : []),
            ...(bounds.top < 15 ? [`top=${bounds.top}`] : []),
            ...(bounds.bottom > 345 ? [`bottom=${bounds.bottom}`] : []),
          ];
        }),
      )
      .toEqual([]);

    await page.setViewportSize({ height: 800, width: 240 });
    await input.focus();
    await page.getByRole('button', { name: DIAGNOSTICS_LABEL }).focus();
    await expect(diagnosticsPopup).toBeVisible();
    await waitForPopupAnimation();
    await expect
      .poll(() =>
        diagnosticsPopup.evaluate((element) => {
          const bounds = element.getBoundingClientRect();

          return [
            ...(bounds.left < 15 ? [`left=${bounds.left}`] : []),
            ...(bounds.right > 225 ? [`right=${bounds.right}`] : []),
            ...(bounds.top < 15 ? [`top=${bounds.top}`] : []),
            ...(bounds.bottom > 785 ? [`bottom=${bounds.bottom}`] : []),
          ];
        }),
      )
      .toEqual([]);
  } finally {
    await chatServer.close();
  }
});

import { expect, test } from '@playwright/test';

import { startChatStreamServer, toolRunChunks } from './helpers/sse';

const RESPONSE_ID = '11111111-2222-3333-4444-555555555555';
const INFERENCE_MODEL = 'claude-sonnet-4-6';
const PREAMBLE = 'Дозволете да проверам…';
const STATUS_LABEL = '🔍 Пребарувам…';
const TOOL = 'search_documents';
const ANSWER = 'Резултатите од испитите се објавуваат на https://finki.ukim.mk';
const LINK_NAME = /finki\.ukim\.mk/u;

test.describe('chat streaming (mocked BFF)', () => {
  test('shows the search chip, drops the preamble, renders the answer, and likes', async ({
    page,
  }) => {
    // The chat stream is served from a tiny local SSE server (not route.fulfill,
    // which delivers atomically) so the transient "searching…" chip gets a real
    // paint before the answer arrives — mirroring how the BFF flushes chunks.
    const chunks = toolRunChunks({
      answer: ANSWER,
      inferenceModel: INFERENCE_MODEL,
      preamble: PREAMBLE,
      responseId: RESPONSE_ID,
      statusLabel: STATUS_LABEL,
      tool: TOOL,
    });
    const statusIndex = chunks.findIndex((c) => c.type === 'data-status');
    const chatServer = await startChatStreamServer({
      gapMs: 600,
      head: chunks.slice(0, statusIndex + 1),
      tail: chunks.slice(statusIndex + 1),
    });

    // 1) models picker -> deterministic list
    await page.route('**/api/models', async (route) => {
      await route.fulfill({
        body: JSON.stringify([INFERENCE_MODEL, 'gpt-5.4-mini']),
        contentType: 'application/json',
        status: 200,
      });
    });

    // 2) chat -> redirect the same-origin POST to the streaming SSE server
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        headers: { location: chatServer.url },
        status: 307,
      });
    });

    // 3) feedback -> capture the request, return an ack
    let feedbackBody: null | Record<string, unknown> = null;
    await page.route('**/api/feedback', async (route) => {
      feedbackBody = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        body: JSON.stringify({
          /* eslint-disable camelcase -- snake_case mirrors the FeedbackAck wire contract */
          feedback_type: 'like',
          id: RESPONSE_ID,
          response_id: RESPONSE_ID,
          /* eslint-enable camelcase -- re-enable past the wire-shaped ack */
        }),
        contentType: 'application/json',
        status: 200,
      });
    });

    await page.goto('/');

    // submit a question via the composer (Macedonian placeholder from i18n)
    const input = page.getByTestId('composer-input');
    await input.fill('Кога се објавуваат резултатите?');
    await input.press('Enter');

    // (1) the search chip appears for the tool call
    const chip = page.getByTestId('search-status');
    await expect(chip).toBeVisible();
    await expect(chip).toContainText('Пребарувам');

    // (3) the final answer renders, and the bare URL is autolinked. Streamdown v2
    // hardens links into a `<button data-streamdown="link">` (not a raw anchor),
    // so the autolinked URL surfaces as a button named after the URL.
    const answer = page.getByTestId('answer-text');
    await expect(answer).toContainText('Резултатите од испитите се објавуваат');
    const autolink = answer.locator('[data-streamdown="link"]', {
      hasText: LINK_NAME,
    });
    await expect(autolink).toBeVisible();

    // (2) the preamble text part was dropped (render-last): it must NOT be on screen
    await expect(page.getByText(PREAMBLE)).toHaveCount(0);

    // (4) like posts to /api/feedback with the response id + feedback type
    await page.getByTestId('like-button').click();
    await expect.poll(() => feedbackBody).not.toBeNull();
    expect(feedbackBody).toMatchObject({
      feedbackType: 'like',
      responseId: RESPONSE_ID,
    });

    await chatServer.close();
  });
});

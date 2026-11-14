import { expect, type Page, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer, type UiChunk } from './helpers/sse';

const MODEL = 'claude-sonnet-5-5';
const ACCOUNT_MENU_NAME_PATTERN = /Корисничко мени:/u;
const ANSWER_TEXT = 'answer-text';
const MORE_QUESTION = 'Каде да видам повеќе?';
const LONG_SAFE_URL = `https://example.org/${'a'.repeat(3_000)}`;

const answer = (text: string): UiChunk[] => [
  {
    messageMetadata: { inferenceModel: MODEL, responseId: 'r1' },
    type: 'start',
  },
  { id: 'a', type: 'text-start' },
  { delta: text, id: 'a', type: 'text-delta' },
  { id: 'a', type: 'text-end' },
  { type: 'finish' },
];

const mockBackend = async (
  page: Page,
  chunks: UiChunk[],
): Promise<{ close: () => Promise<void> }> => {
  const server = await startChatStreamServer({
    gapMs: 30,
    head: chunks.slice(0, 1),
    tail: chunks.slice(1),
  });
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: '{}',
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page);
  await installMockChatState(page, { streamUrl: server.url });
  return server;
};

const send = async (page: Page, text: string): Promise<void> => {
  const input = page.getByTestId('composer-input');
  await input.fill(text);
  const submit = page.getByTestId('composer-submit');
  await expect(submit).toBeEnabled();
  await submit.click();
};

test('clicking a link opens the shared confirmation modal', async ({
  page,
}) => {
  const server = await mockBackend(
    page,
    answer('Повеќе на https://example.org тука.'),
  );
  await page.goto('/');
  await send(page, MORE_QUESTION);

  const link = page
    .getByTestId(ANSWER_TEXT)
    .locator('[data-streamdown="link"]')
    .first();
  await link.waitFor();
  await link.click();

  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('Отвори надворешна врска?');
  await expect(dialog).toContainText('example.org');

  await page.getByRole('button', { name: 'Откажи' }).click();
  await expect(page.getByRole('dialog')).toHaveCount(0);

  await server.close();
});

test('faculty links open directly, without the confirmation', async ({
  page,
}) => {
  const server = await mockBackend(
    page,
    answer('Повеќе на https://finki.ukim.mk тука.'),
  );
  await page.goto('/');
  await send(page, MORE_QUESTION);

  const link = page
    .getByTestId(ANSWER_TEXT)
    .locator('[data-streamdown="link"]')
    .first();
  await link.waitFor();
  await page.context().route('https://finki.ukim.mk/**', async (route) => {
    await route.fulfill({ body: 'ok', status: 200 });
  });
  const popupEvent = page.waitForEvent('popup');
  await link.click();
  const popup = await popupEvent;

  expect(popup.url()).toBe('https://finki.ukim.mk/');
  await expect(page.getByRole('dialog')).toHaveCount(0);

  await server.close();
});

test('protocol-less markdown links render as safe https links', async ({
  page,
}) => {
  const server = await mockBackend(
    page,
    answer(
      'Пример:\n\n```md\nUse [FINKI](example.org)\n```\n\nПовеќе на [FINKI](example.org) тука.',
    ),
  );
  await page.goto('/');
  await send(page, MORE_QUESTION);

  const answerText = page.getByTestId(ANSWER_TEXT);
  await expect(answerText).not.toContainText('[blocked]');

  const link = answerText.locator('[data-streamdown="link"]', {
    hasText: 'FINKI',
  });
  await expect(link).toBeVisible();

  const code = answerText.locator('code', {
    hasText: 'Use [FINKI](example.org)',
  });
  await expect(code).toBeVisible();
  await expect(code).not.toContainText('https://example.org');

  await link.click();

  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('https://example.org/');

  await server.close();
});

test('long link confirmations scroll inside mobile safe areas', async ({
  page,
}) => {
  // Given a mobile viewport with cutouts and an answer containing a long safe URL.
  await page.setViewportSize({ height: 844, width: 390 });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setSafeAreaInsetsOverride', {
    insets: { bottom: 34, left: 47, right: 21, top: 59 },
  });
  const server = await mockBackend(
    page,
    answer(`Повеќе на [FINKI](${LONG_SAFE_URL}) тука.`),
  );
  await page.goto('/');
  await send(page, MORE_QUESTION);

  // When the user opens the link confirmation.
  const link = page
    .getByTestId(ANSWER_TEXT)
    .locator('[data-streamdown="link"]')
    .first();
  await link.waitFor();
  await link.click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();

  // Then the generic dialog owns overflow and remains inside the safe viewport.
  await expect(dialog).toHaveCSS('overflow-y', 'auto');
  const metrics = await dialog.evaluate((element) => ({
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
  }));
  expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight);
  await dialog.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  await expect
    .poll(async () => dialog.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(0);

  await server.close();
});

test('delete-all clears the entire history after confirmation', async ({
  page,
}) => {
  const server = await mockBackend(page, answer('Готово.'));
  await page.route('**/api/auth/session', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        expires: '2099-01-01T00:00:00.000Z',
        user: { email: 'student@example.com', name: 'Student' },
      }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.goto('/');
  await send(page, 'Прашање');
  await expect(page.getByTestId(ANSWER_TEXT)).toContainText('Готово');

  const items = page.locator('[data-testid^="conversation-"]');
  await expect(items).toHaveCount(1);

  await page.getByRole('button', { name: ACCOUNT_MENU_NAME_PATTERN }).click();
  await page
    .getByRole('menuitem', { name: 'Избриши ги сите разговори' })
    .click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toContainText('Избриши ги сите разговори?');

  await page.getByTestId('confirm-action').click();
  await expect(items).toHaveCount(0);

  await server.close();
});

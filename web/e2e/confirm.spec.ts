import { expect, type Page, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer, type UiChunk } from './helpers/sse';

const MODEL = 'claude-sonnet-5';

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
  await mockModels(page);
  await installMockChatState(page, { streamUrl: server.url });
  return server;
};

const send = async (page: Page, text: string): Promise<void> => {
  const input = page.getByTestId('composer-input');
  await input.fill(text);
  await input.press('Enter');
};

test('clicking a link opens the shared confirmation modal', async ({
  page,
}) => {
  const server = await mockBackend(
    page,
    answer('Повеќе на https://finki.ukim.mk тука.'),
  );
  await page.goto('/');
  await send(page, 'Каде да видам повеќе?');

  const link = page
    .getByTestId('answer-text')
    .locator('[data-streamdown="link"]')
    .first();
  await link.waitFor();
  await link.click();

  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('Отвори надворешна врска?');
  await expect(dialog).toContainText('finki.ukim.mk');

  await page.getByRole('button', { name: 'Откажи' }).click();
  await expect(page.getByRole('dialog')).toHaveCount(0);

  await server.close();
});

test('protocol-less markdown links render as safe https links', async ({
  page,
}) => {
  const server = await mockBackend(
    page,
    answer(
      'Пример:\n\n```md\nUse [FINKI](finki.ukim.mk)\n```\n\nПовеќе на [FINKI](finki.ukim.mk) тука.',
    ),
  );
  await page.goto('/');
  await send(page, 'Каде да видам повеќе?');

  const answerText = page.getByTestId('answer-text');
  await expect(answerText).not.toContainText('[blocked]');

  const link = answerText.locator('[data-streamdown="link"]', {
    hasText: 'FINKI',
  });
  await expect(link).toBeVisible();

  const code = answerText.locator('code', {
    hasText: 'Use [FINKI](finki.ukim.mk)',
  });
  await expect(code).toBeVisible();
  await expect(code).not.toContainText('https://finki.ukim.mk');

  await link.click();

  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('https://finki.ukim.mk/');

  await server.close();
});

test('delete-all clears the entire history after confirmation', async ({
  page,
}) => {
  const server = await mockBackend(page, answer('Готово.'));
  await page.goto('/');
  await send(page, 'Прашање');
  await expect(page.getByTestId('answer-text')).toContainText('Готово');

  const items = page.locator('[data-testid^="conversation-"]');
  await expect(items).toHaveCount(1);

  await page.getByTestId('delete-all').click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toContainText('Избриши ги сите разговори?');

  await page.getByTestId('confirm-action').click();
  await expect(items).toHaveCount(0);

  await server.close();
});

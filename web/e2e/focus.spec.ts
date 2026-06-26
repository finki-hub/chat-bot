import { expect, type Page, test } from '@playwright/test';

import { startChatStreamServer, type UiChunk } from './helpers/sse';

const MODEL = 'claude-sonnet-4-6';

const chunks: UiChunk[] = [
  {
    messageMetadata: { inferenceModel: MODEL, responseId: 'r1' },
    type: 'start',
  },
  { id: 'a', type: 'text-start' },
  { delta: 'Готово.', id: 'a', type: 'text-delta' },
  { id: 'a', type: 'text-end' },
  { type: 'finish' },
];

const mockBackend = async (
  page: Page,
): Promise<{ close: () => Promise<void> }> => {
  const server = await startChatStreamServer({
    gapMs: 20,
    head: chunks.slice(0, 1),
    tail: chunks.slice(1),
  });
  await page.route('**/api/models', async (route) => {
    await route.fulfill({
      body: JSON.stringify([MODEL]),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.route('**/api/chat', async (route) => {
    await route.fulfill({ headers: { location: server.url }, status: 307 });
  });
  return server;
};

test('composer is focused on load and after a response finishes', async ({
  page,
}) => {
  const server = await mockBackend(page);
  await page.goto('/');

  const input = page.getByTestId('composer-input');
  await expect(input).toBeFocused();

  await input.fill('Прашање');
  await input.press('Enter');
  await expect(page.getByTestId('answer-text')).toContainText('Готово');

  await expect(input).toBeFocused();

  await server.close();
});

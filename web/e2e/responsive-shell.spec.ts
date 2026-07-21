import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const STREAM_URL = 'http://127.0.0.1:9/stream';
const CONVERSATION_TITLE = 'Услови за запишување семестар';

test('tablet width keeps the chat sidebar in a modal drawer', async ({
  page,
}) => {
  // Given a tablet-width chat shell.
  await page.setViewportSize({ height: 900, width: 768 });
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page);
  await installMockChatState(page, {
    conversations: [
      { id: 'conversation-1', model: null, title: CONVERSATION_TITLE },
    ],
    streamUrl: STREAM_URL,
  });
  await page.goto('/');

  // When the user opens navigation.
  await expect(page.getByRole('complementary')).toHaveCount(0);
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();

  // Then navigation overlays the chat instead of permanently narrowing it.
  await expect(
    page.getByRole('dialog', { name: 'Странична лента' }),
  ).toBeVisible();
  await expect(page.locator('#main-content')).toHaveCSS('width', '768px');

  await page.getByRole('button', { name: CONVERSATION_TITLE }).click();

  await expect(
    page.getByRole('dialog', { name: 'Странична лента' }),
  ).toBeHidden();
});

test('narrow fine pointers retain inline conversation actions', async ({
  page,
}) => {
  // Given a narrow desktop browser with a fine primary pointer.
  await page.setViewportSize({ height: 812, width: 375 });
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page);
  await installMockChatState(page, {
    conversations: [
      { id: 'conversation-1', model: null, title: CONVERSATION_TITLE },
    ],
    streamUrl: STREAM_URL,
  });
  await page.goto('/');
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const row = page.getByTestId('conversation-conversation-1');

  // When the fine pointer hovers the conversation row.
  await row.hover();

  // Then inline controls are available without the coarse-pointer overflow trigger.
  await expect(row.getByTestId('row-actions')).toHaveCSS('display', 'flex');
  await expect(
    row.getByRole('button', {
      name: `Дејства за разговорот: ${CONVERSATION_TITLE}`,
    }),
  ).toHaveCount(0);
  await expect(row.getByRole('button', { name: 'Преименувај' })).toBeVisible();
});

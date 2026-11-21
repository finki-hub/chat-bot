import { expect, type Page, test } from '@playwright/test';

import type { ChatConversationHistory } from '@/lib/conversation-types';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const STREAM_URL = 'http://127.0.0.1:9/stream';
const TARGET_MESSAGE_ID = 'long-message-12';
const RESULT_TITLE_PATTERN = /Рокови за испити/u;
const HIGHLIGHT_PATTERN = /ring-primary\/50/u;

const history = (
  conversationId: string,
  title: string,
  messages: ChatConversationHistory['messages'],
): ChatConversationHistory => ({
  conversation: { activeStream: null, id: conversationId, model: null, title },
  messages,
});

const shortConversation = history('short', 'Краток разговор', [
  {
    id: 'short-message-1',
    parts: [{ text: 'Порака од краткиот разговор', type: 'text' }],
    role: 'user',
  },
]);

const longConversation = history(
  'long',
  'Рокови за испити',
  Array.from({ length: 40 }, (_, index) => ({
    id: `long-message-${index}`,
    parts: [
      {
        text:
          index === 12
            ? 'Менторот рече дека испитот е закажан за јуни.'
            : `Порака ${index} со доволно текст за да има лизгање.`,
        type: 'text' as const,
      },
    ],
    role: index % 2 === 0 ? ('user' as const) : ('assistant' as const),
  })),
);

const setup = async (page: Page) => {
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
  await mockModels(page);
  await installMockChatState(page, {
    conversations: [
      { id: 'short', model: null, title: 'Краток разговор' },
      { id: 'long', model: null, title: 'Рокови за испити' },
    ],
    histories: { long: longConversation, short: shortConversation },
    streamUrl: STREAM_URL,
  });
  await page.route('**/api/chat/search?*', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        query: 'ментор испит',
        results: [
          {
            conversationId: 'long',
            conversationTitle: 'Рокови за испити',
            conversationUpdatedAt: '2026-11-02T12:00:00Z',
            message: {
              createdAt: '2026-11-02T12:00:00Z',
              id: TARGET_MESSAGE_ID,
              role: 'assistant',
            },
            snippet:
              '\u{1}Менторот\u{2} рече дека \u{1}испитот\u{2} е закажан за јуни.',
          },
        ],
      }),
      contentType: 'application/json',
      status: 200,
    });
  });
};

test('a search result opens another conversation at the matched message', async ({
  page,
}) => {
  await page.setViewportSize({ height: 800, width: 1_280 });
  await setup(page);
  const historyGate = Promise.withResolvers<undefined>();
  await page.route('**/api/chat/long/history', async (route) => {
    await historyGate.promise;
    await route.fulfill({
      body: JSON.stringify(longConversation),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.goto('/');
  await page
    .getByRole('button', { exact: true, name: 'Краток разговор' })
    .click();
  await expect(page.getByText('Порака од краткиот разговор')).toBeVisible();

  await page.getByRole('searchbox').fill('ментор испит');
  const result = page
    .getByTestId('conversation-search-results')
    .getByRole('button', { name: RESULT_TITLE_PATTERN });
  await expect(result).toContainText('Асистент');
  await page.screenshot({ path: 'test-results/conversation-search.png' });
  await result.click();

  // The previous conversation is still shown while the history loads.
  await expect(page.getByText('Порака од краткиот разговор')).toBeVisible();
  historyGate.resolve(undefined);

  const target = page.locator(`[data-message-id="${TARGET_MESSAGE_ID}"]`);
  await expect(target).toHaveClass(HIGHLIGHT_PATTERN);
  await expect(target).toBeInViewport();
});

test('Ctrl+K opens a closed sidebar and focuses search', async ({ page }) => {
  await page.setViewportSize({ height: 800, width: 1_280 });
  await setup(page);
  await page.goto('/');
  const search = page.getByRole('searchbox');
  await expect(search).toBeVisible();
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await expect(search).toBeHidden();

  await page.keyboard.press('Control+K');

  await expect(search).toBeFocused();
});

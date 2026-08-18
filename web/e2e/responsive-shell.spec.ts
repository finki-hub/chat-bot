import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const STREAM_URL = 'http://127.0.0.1:9/stream';
const CONVERSATION_TITLE = 'Услови за запишување семестар';
const MINIMUM_SHORT_THREAD_HEIGHT = 160;
const VIEWPORT_FIT_COVER_PATTERN = /viewport-fit=cover/u;
const DISCLAIMER_PATTERN = /Не внесувајте лични/u;
const LONG_HISTORY = Array.from({ length: 12 }, (_, index) => ({
  id: `message-${index}`,
  parts: [
    {
      text: `Долга тест порака ${index} со доволно текст за прелевање.`,
      type: 'text' as const,
    },
  ],
  role: index % 2 === 0 ? ('user' as const) : ('assistant' as const),
}));

const mockSession = async (page: Parameters<typeof mockModels>[0]) => {
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
};

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
  await mockSession(page);
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
  await mockSession(page);
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

test('short mobile viewports preserve a readable message region', async ({
  page,
}) => {
  // Given a keyboard-height viewport with an active conversation.
  await page.setViewportSize({ height: 430, width: 320 });
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, {
    conversations: [
      { id: 'conversation-1', model: null, title: CONVERSATION_TITLE },
    ],
    histories: {
      'conversation-1': {
        conversation: {
          activeStream: null,
          id: 'conversation-1',
          model: null,
          title: CONVERSATION_TITLE,
        },
        messages: LONG_HISTORY,
      },
    },
    streamUrl: STREAM_URL,
  });
  await page.goto('/');
  await expect(page.locator('meta[name="viewport"]')).toHaveAttribute(
    'content',
    VIEWPORT_FIT_COVER_PATTERN,
  );
  await expect(
    page.getByRole('log').getByRole('img', { name: 'ФИНКИ Хаб' }),
  ).toBeInViewport({ ratio: 1 });
  await expect(
    page.getByRole('button', { name: 'Скролувај до најновата порака' }),
  ).toHaveCount(0);
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await page.getByRole('button', { name: CONVERSATION_TITLE }).click();
  await expect(
    page.getByText('Долга тест порака 11 со доволно текст за прелевање.'),
  ).toBeAttached();

  // When the fixed shell settles, the conversation remains practically usable.
  const thread = await page.getByRole('log').boundingBox();
  const contextBar = page.getByTestId('chat-context-bar');
  const disclaimer = page.getByText(DISCLAIMER_PATTERN);

  // Then fixed chrome leaves enough height to read more than one message line.
  expect(thread?.height).toBeGreaterThanOrEqual(MINIMUM_SHORT_THREAD_HEIGHT);
  await expect(contextBar).toHaveCSS('min-height', '44px');
  await expect
    .poll(async () =>
      disclaimer.evaluate(
        (element) => getComputedStyle(element).webkitLineClamp,
      ),
    )
    .toBe('1');
  await expect
    .poll(async () =>
      page
        .getByRole('log')
        .locator(':scope > div')
        .first()
        .evaluate(
          (element) =>
            element.scrollHeight - element.clientHeight - element.scrollTop,
        ),
    )
    .toBeLessThanOrEqual(1);
});

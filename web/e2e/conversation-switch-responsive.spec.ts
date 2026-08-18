import { expect, test } from '@playwright/test';

import type { ChatConversationHistory } from '@/lib/conversation-types';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const STREAM_URL = 'http://127.0.0.1:9/stream';

const history = (
  conversationId: string,
  title: string,
  messages: ChatConversationHistory['messages'],
): ChatConversationHistory => ({
  conversation: {
    activeStream: null,
    id: conversationId,
    model: null,
    title,
  },
  messages,
});

test('switching populated conversations opens the new history at the latest message', async ({
  page,
}) => {
  // Given a short conversation and a much longer conversation whose history is gated.
  await page.setViewportSize({ height: 430, width: 320 });
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

  const conversationA = history('conversation-a', 'Краток разговор', [
    {
      id: 'conversation-a-message-1',
      parts: [{ text: 'Порака од разговор А', type: 'text' }],
      role: 'user',
    },
  ]);
  const conversationB = history(
    'conversation-b',
    'Долг разговор',
    Array.from({ length: 60 }, (_, index) => ({
      id: `conversation-b-message-${index}`,
      parts: [
        {
          text: `Порака од разговор Б ${index} со текст за прелевање.`,
          type: 'text' as const,
        },
      ],
      role: index % 2 === 0 ? ('user' as const) : ('assistant' as const),
    })),
  );
  await installMockChatState(page, {
    conversations: [
      { id: 'conversation-a', model: null, title: 'Краток разговор' },
      { id: 'conversation-b', model: null, title: 'Долг разговор' },
    ],
    histories: {
      'conversation-a': conversationA,
      'conversation-b': conversationB,
    },
    streamUrl: STREAM_URL,
  });
  const historyGate = Promise.withResolvers<undefined>();
  await page.route('**/api/chat/conversation-b/history', async (route) => {
    await historyGate.promise;
    await route.fulfill({
      body: JSON.stringify(conversationB),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.goto('/');
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await page.getByRole('button', { name: 'Краток разговор' }).click();
  await expect(page.getByText('Порака од разговор А')).toBeVisible();

  // When the user selects the longer populated conversation.
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await page.getByRole('button', { name: 'Долг разговор' }).click();
  await expect(page.getByText('Порака од разговор А')).toBeVisible();
  historyGate.resolve(undefined);
  await expect(
    page.getByText('Порака од разговор Б 59 со текст за прелевање.'),
  ).toBeAttached();

  // Then the newly hydrated history is already positioned at its latest message.
  const distanceFromBottom = await page
    .getByRole('log')
    .locator(':scope > div')
    .first()
    .evaluate(
      (element) =>
        element.scrollHeight - element.clientHeight - element.scrollTop,
    );
  expect(distanceFromBottom).toBeLessThanOrEqual(1);
});

import type { Page, Route } from '@playwright/test';

type ConversationRow = {
  readonly id: string;
  readonly model: null | string;
  readonly title: string;
};

type MockChatStateOptions = {
  readonly streamUrl: string;
};

const emptyHistory = (id: string) => ({
  conversation: { id, model: null, title: 'New conversation' },
  messages: [],
});

const conversationIdFrom = (route: Route): string => {
  const pathname = new URL(route.request().url()).pathname;
  return decodeURIComponent(pathname.split('/', 4)[3] ?? 'conversation');
};

export const installMockChatState = async (
  page: Page,
  { streamUrl }: MockChatStateOptions,
): Promise<void> => {
  const conversations: ConversationRow[] = [];

  await page.route('**/api/chat/*/history', async (route) => {
    await route.fulfill({
      body: JSON.stringify(emptyHistory(conversationIdFrom(route))),
      contentType: 'application/json',
      status: 200,
    });
  });

  await page.route('**/api/chat/*/stop', async (route) => {
    await route.fulfill({ status: 204 });
  });

  await page.route('**/api/chat/*', async (route) => {
    const method = route.request().method();
    const id = conversationIdFrom(route);

    if (method === 'PATCH') {
      if (conversations.every((conversation) => conversation.id !== id)) {
        conversations.unshift({ id, model: null, title: 'New conversation' });
      }
      await route.fulfill({ status: 204 });
      return;
    }

    if (method === 'DELETE') {
      const index = conversations.findIndex(
        (conversation) => conversation.id === id,
      );
      if (index !== -1) {
        conversations.splice(index, 1);
      }
      await route.fulfill({ status: 204 });
      return;
    }

    await route.fallback();
  });

  await page.route('**/api/chat', async (route) => {
    const method = route.request().method();

    if (method === 'POST') {
      await route.fulfill({ headers: { location: streamUrl }, status: 307 });
      return;
    }

    if (method === 'GET') {
      await route.fulfill({
        body: JSON.stringify(conversations),
        contentType: 'application/json',
        status: 200,
      });
      return;
    }

    if (method === 'DELETE') {
      conversations.length = 0;
      await route.fulfill({
        body: JSON.stringify([]),
        contentType: 'application/json',
        status: 200,
      });
      return;
    }

    await route.fallback();
  });
};

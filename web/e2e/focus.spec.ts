import { expect, type Locator, type Page, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer, type UiChunk } from './helpers/sse';

const MODEL = 'claude-sonnet-5';

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
  await mockModels(page);
  await installMockChatState(page, { streamUrl: server.url });
  return server;
};

const tooltipTriggerFor = (control: Locator): Locator =>
  control.locator('xpath=ancestor-or-self::*[@data-slot="tooltip-trigger"][1]');

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

test('answer actions explain their actions on hover and focus', async ({
  page,
}) => {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  const server = await mockBackend(page);
  await page.goto('/');

  const input = page.getByTestId('composer-input');
  await input.fill('Прашање');
  await input.press('Enter');
  await expect(page.getByTestId('answer-text')).toContainText('Готово');

  const answer = page.getByTestId('answer-text');
  const actions = page.getByTestId('answer-actions');
  const tooltip = page.getByRole('tooltip');
  const controls = [
    'Копирај',
    'Регенерирај',
    'Ми се допаѓа',
    'Не ми се допаѓа',
  ].map((label) => ({
    control: actions.getByRole('button', { exact: true, name: label }),
    label,
  }));

  for (const { control, label } of controls) {
    await tooltipTriggerFor(control).hover();
    await expect(tooltip).toHaveText(label);
    await answer.hover();
    await expect(tooltip).toBeHidden();
  }

  for (const { control, label } of controls) {
    await control.focus();
    await expect(tooltip).toHaveText(label);
    await page.keyboard.press('Escape');
    await expect(tooltip).toBeHidden();
  }

  await server.close();
});

test('header controls explain their actions on hover and focus', async ({
  page,
}) => {
  await page.route('**/api/auth/session', async (route) => {
    await route.fulfill({
      body: 'null',
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page);
  await installMockChatState(page, {
    streamUrl: 'http://127.0.0.1:9/stream',
  });
  await page.goto('/');

  const githubControl = page.getByRole('link', {
    name: 'GitHub репозиториум',
  });
  const themeControl = page.getByRole('button', { name: 'Промени тема' });
  const headerTitle = page.getByRole('heading', { name: 'ФИНКИ Хаб' });
  const tooltip = page.getByRole('tooltip');

  await expect(
    page.getByRole('button', { name: 'Сподели разговор' }),
  ).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'API клучеви' })).toHaveCount(
    0,
  );
  await expect(page.getByRole('button', { name: 'Најави се' })).toHaveCount(0);

  const controls = [
    {
      control: githubControl,
      label: 'GitHub репозиториум',
    },
    {
      control: themeControl,
      label: 'Промени тема',
    },
  ];

  for (const { control, label } of controls) {
    await tooltipTriggerFor(control).hover();
    await expect(tooltip).toHaveText(label);
    await headerTitle.hover();
    await expect(tooltip).toBeHidden();
  }

  for (const { control, label } of controls) {
    await expect(control).toBeEnabled();
    await control.focus();
    await expect(tooltip).toHaveText(label);
    await page.keyboard.press('Escape');
    await expect(tooltip).toBeHidden();
  }
});

test('sidebar conversation actions explain their actions on hover and focus', async ({
  page,
}) => {
  await mockModels(page);
  await installMockChatState(page, {
    conversations: [
      {
        id: 'c1',
        model: MODEL,
        title: 'Прв разговор',
      },
    ],
    streamUrl: 'http://127.0.0.1:9/stream',
  });
  await page.goto('/');

  const item = page.getByTestId('conversation-c1');
  const conversationTitle = item.getByRole('button', { name: 'Прв разговор' });
  const tooltip = page.getByRole('tooltip');
  const controls = ['Генерирај име', 'Преименувај', 'Избриши'].map((label) => ({
    control: item.getByRole('button', { name: label }),
    label,
  }));

  for (const { control, label } of controls) {
    await tooltipTriggerFor(control).hover();
    await expect(tooltip).toHaveText(label);
    await conversationTitle.hover();
    await expect(tooltip).toBeHidden();
  }

  for (const { control, label } of controls) {
    await control.focus();
    await expect(tooltip).toHaveText(label);
    await page.keyboard.press('Escape');
    await expect(tooltip).toBeHidden();
  }
});

import { expect, type Page, test } from '@playwright/test';

import type { ModelCatalog } from '@/lib/api-types';

/* eslint-disable camelcase -- catalog fixtures mirror the API wire contract. */
import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

type MockSessionUser = {
  readonly email?: string;
  readonly name?: string;
};

const DEFAULT_SESSION_USER = {
  email: 'student@example.com',
  name: 'Student',
} as const satisfies MockSessionUser;
const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const STREAM_URL = 'http://127.0.0.1:9/stream';
const SPONSORED_MODEL = 'gpt-5.6-luna';
const SPONSORED_CATALOG: ModelCatalog = {
  models: [
    {
      availability: 'sponsored',
      id: SPONSORED_MODEL,
      name: 'GPT-5.6 Luna',
      provider: 'openai',
      sponsored_quota: {
        limit: 5,
        remaining: 3,
        resets_at: '2099-01-01T12:00:00Z',
      },
    },
  ],
  source: 'live',
  version: 1,
};
const mockSession = async (
  page: Page,
  user: MockSessionUser = DEFAULT_SESSION_USER,
): Promise<void> => {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.route('**/api/auth/session', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        expires: '2099-01-01T00:00:00.000Z',
        user,
      }),
      contentType: 'application/json',
      status: 200,
    });
  });
};

test('mobile drawer traps focus, closes with Escape, and restores the trigger', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const trigger = page.getByRole('button', {
    name: SIDEBAR_TOGGLE_LABEL,
  });
  await trigger.click();

  const drawer = page.getByRole('dialog', { name: 'Странична лента' });
  await expect(drawer).toBeVisible();
  await expect(
    drawer.getByRole('button', { name: 'Нов разговор' }),
  ).toBeFocused();

  await page.keyboard.press('Shift+Tab');
  await expect(drawer).toContainText('Нов разговор');

  await page.keyboard.press('Escape');
  await expect(drawer).toBeHidden();
  await expect(trigger).toBeFocused();
});

test('mobile primary controls meet the 44px touch target minimum', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const controls = [
    page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }),
    page.getByRole('button', { name: 'Промени тема' }),
    page.getByTestId('composer-submit'),
  ];

  for (const control of controls) {
    const box = await control.boundingBox();
    expect(box?.height).toBeGreaterThanOrEqual(44);
    expect(box?.width).toBeGreaterThanOrEqual(44);
  }

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const drawer = page.getByRole('dialog', { name: 'Странична лента' });
  const accountTrigger = drawer.getByRole('button', {
    name: 'Корисничко мени: Student, student@example.com',
  });
  const triggerBox = await accountTrigger.boundingBox();

  expect(triggerBox?.height).toBeGreaterThanOrEqual(44);
  expect(triggerBox?.width).toBeGreaterThanOrEqual(44);
});

test('mobile keeps account actions in the profile menu and opens credentials after closing the drawer', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const drawer = page.getByRole('dialog', { name: 'Странична лента' });
  const accountTrigger = drawer.getByRole('button', {
    name: 'Корисничко мени: Student, student@example.com',
  });

  await expect(accountTrigger).toContainText('Student');
  await expect(accountTrigger).toContainText('student@example.com');
  await accountTrigger.click();
  await expect(page.getByRole('menuitem', { name: 'Одјави се' })).toBeVisible();
  await page.getByRole('menuitem', { name: 'API клучеви' }).click();

  await expect(drawer).toBeHidden();
  const credentialsDialog = page.getByRole('dialog', {
    name: 'Лични API клучеви',
  });
  await expect(credentialsDialog).toBeVisible();
  await expect(page.getByRole('button', { name: 'API клучеви' })).toHaveCount(
    0,
  );
  await expect(page.getByRole('button', { name: 'Одјави се' })).toHaveCount(0);
});

test('mobile drawer truncates long authenticated names without overflowing', async ({
  page,
}) => {
  const name = 'Student With An Exceptionally Long Display Name';
  await page.setViewportSize({ height: 812, width: 320 });
  await mockSession(page, { email: 'student@example.com', name });
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const identity = page.getByTestId('sidebar-user-identity');
  const label = identity.getByText(name);
  await expect(label).toBeVisible();
  await expect(label).toHaveCSS('overflow', 'hidden');
  await expect(label).toHaveCSS('text-overflow', 'ellipsis');
  await expect(label).toHaveCSS('white-space', 'nowrap');
  await expect
    .poll(async () =>
      label.evaluate((element) => element.scrollWidth > element.clientWidth),
    )
    .toBe(true);
  const box = await identity.boundingBox();
  expect((box?.x ?? Infinity) + (box?.width ?? Infinity)).toBeLessThanOrEqual(
    320,
  );
});

test('mobile keeps the account menu available when authenticated identity is unavailable', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 320 });
  await mockSession(page, {});
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  await expect(page.getByRole('button', { name: 'Одјави се' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'API клучеви' })).toHaveCount(
    0,
  );

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const accountTrigger = page.getByRole('button', {
    name: 'Корисничко мени: Сметка',
  });

  await expect(accountTrigger).toContainText('Сметка');
  await accountTrigger.click();
  await expect(
    page.getByRole('menuitem', { name: 'API клучеви' }),
  ).toBeVisible();
  await expect(page.getByRole('menuitem', { name: 'Одјави се' })).toBeVisible();
});

test('mobile model selector exposes the sponsored badge and remaining quota', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page, {
    catalog: SPONSORED_CATALOG,
    credentialProviders: [],
  });
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const trigger = page.getByTestId('composer-model');
  await expect(trigger).toContainText('GPT-5.6 Luna');
  await trigger.click();

  await expect(page.getByTestId('model-provider-label')).toHaveText('OpenAI');
  const badge = page.getByTestId(`model-free-badge-${SPONSORED_MODEL}`);
  await expect(badge).toContainText('Бесплатно');
  await expect(badge).toContainText('3/5');
  await page.screenshot({
    animations: 'disabled',
    path: testInfo.outputPath('sponsored-selector-mobile.png'),
  });
});

/* eslint-enable camelcase -- end catalog wire fixtures. */

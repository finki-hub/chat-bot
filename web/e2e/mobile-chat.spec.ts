import { expect, type Page, test } from '@playwright/test';

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
const STREAM_URL = 'http://127.0.0.1:9/stream';

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
    name: 'Прикажи/сокриј странична лента',
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
    page.getByRole('button', { name: 'Прикажи/сокриј странична лента' }),
    page.getByRole('button', { name: 'API клучеви' }),
    page.getByRole('button', { name: 'Одјави се: Student' }),
    page.getByRole('button', { name: 'Промени тема' }),
    page.getByTestId('composer-submit'),
  ];

  for (const control of controls) {
    const box = await control.boundingBox();
    expect(box?.height).toBeGreaterThanOrEqual(44);
    expect(box?.width).toBeGreaterThanOrEqual(44);
  }
});

test('mobile authenticated header shows the current user name', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const signOutButton = page.getByRole('button', {
    name: 'Одјави се: Student',
  });
  await expect(signOutButton.getByText('Student')).toBeVisible();

  const box = await signOutButton.boundingBox();
  expect(box?.width).toBeGreaterThan(box?.height ?? Infinity);
});

test('mobile authenticated header truncates long names without overflowing', async ({
  page,
}) => {
  const name = 'Student With An Exceptionally Long Display Name';
  await page.setViewportSize({ height: 812, width: 320 });
  await mockSession(page, { email: 'student@example.com', name });
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const signOutButton = page.getByRole('button', {
    name: `Одјави се: ${name}`,
  });
  const label = signOutButton.getByText(name);
  const title = page.getByRole('heading', { name: 'ФИНКИ Хаб' });
  await expect(label).toBeVisible();
  await expect(label).toHaveCSS('overflow', 'hidden');
  await expect(label).toHaveCSS('text-overflow', 'ellipsis');
  await expect(label).toHaveCSS('white-space', 'nowrap');
  await expect
    .poll(async () =>
      label.evaluate((element) => element.scrollWidth > element.clientWidth),
    )
    .toBe(true);
  await expect
    .poll(async () =>
      title.evaluate((element) => element.scrollWidth <= element.clientWidth),
    )
    .toBe(true);

  const box = await signOutButton.boundingBox();
  expect((box?.x ?? Infinity) + (box?.width ?? Infinity)).toBeLessThanOrEqual(
    320,
  );
});

test('mobile authenticated header shows the full sign-out label without identity', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 320 });
  await mockSession(page, {});
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const signOutButton = page.getByRole('button', {
    exact: true,
    name: 'Одјави се',
  });
  const label = signOutButton.getByText('Одјави се');

  await expect(label).toBeVisible();
  await expect
    .poll(async () =>
      label.evaluate((element) => element.scrollWidth <= element.clientWidth),
    )
    .toBe(true);

  const box = await signOutButton.boundingBox();

  expect(box?.height).toBeGreaterThanOrEqual(44);
  expect(box?.width).toBeGreaterThanOrEqual(44);
});

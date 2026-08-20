import { expect, type Page, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const LEFT_SAFE_AREA = 47;
const RIGHT_SAFE_AREA = 21;
const TOP_SAFE_AREA = 59;
const BOTTOM_SAFE_AREA = 34;
const SAFE_GUTTER = 16;
const ROUNDING_TOLERANCE = 1;
const VIEWPORT_HEIGHT = 844;
const NARROW_VIEWPORT_WIDTH = 390;
const WIDE_VIEWPORT_WIDTH = 640;
const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const ACCOUNT_MENU_LABEL = /Корисничко мени:/u;

const drawerAccountMenu = (page: Page) =>
  page
    .getByRole('dialog', { name: 'Странична лента' })
    .getByRole('button', { name: ACCOUNT_MENU_LABEL });

const prepareSafeAreaPage = async (
  page: Page,
  width: number,
): Promise<void> => {
  await page.setViewportSize({ height: VIEWPORT_HEIGHT, width });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setSafeAreaInsetsOverride', {
    insets: {
      bottom: BOTTOM_SAFE_AREA,
      left: LEFT_SAFE_AREA,
      right: RIGHT_SAFE_AREA,
      top: TOP_SAFE_AREA,
    },
  });
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
    streamUrl: 'http://127.0.0.1:9/stream',
  });
  await page.goto('/');
};

test('sign-in content stays inside vertical display cutouts', async ({
  page,
}) => {
  // Given a signed-out mobile viewport with asymmetric vertical cutouts.
  await page.setViewportSize({
    height: VIEWPORT_HEIGHT,
    width: NARROW_VIEWPORT_WIDTH,
  });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setSafeAreaInsetsOverride', {
    insets: {
      bottom: BOTTOM_SAFE_AREA,
      left: 0,
      right: 0,
      top: TOP_SAFE_AREA,
    },
  });

  // When the sign-in page renders.
  await page.goto('/signin');
  const content = page.locator('#main-content > div').first();
  const padding = await content.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      bottom: Number.parseFloat(style.paddingBottom),
      top: Number.parseFloat(style.paddingTop),
    };
  });

  // Then its content clears both cutouts without reducing baseline spacing.
  expect(padding.top).toBeGreaterThanOrEqual(TOP_SAFE_AREA);
  expect(padding.bottom).toBe(40);
  await page.keyboard.press('Tab');
  const skipLink = page.getByRole('link', {
    name: 'Прескокни до содржината',
  });
  await expect(skipLink).toBeFocused();
  await expect(skipLink).toBeVisible();
  const skipLinkBox = await skipLink.boundingBox();
  expect(skipLinkBox?.y).toBeGreaterThanOrEqual(TOP_SAFE_AREA);
});

test('mobile controls and overlays stay outside display cutouts', async ({
  page,
}) => {
  // Given a mobile viewport with asymmetric safe-area insets on every edge.
  await prepareSafeAreaPage(page, NARROW_VIEWPORT_WIDTH);

  const skipLink = page.getByRole('link', {
    name: 'Прескокни до содржината',
  });
  await skipLink.focus();
  await expect
    .poll(async () => {
      const bounds = await skipLink.boundingBox();
      return bounds?.y ?? -Infinity;
    })
    .toBeGreaterThanOrEqual(TOP_SAFE_AREA);
  const skipLinkBounds = await skipLink.boundingBox();
  expect(skipLinkBounds).not.toBeNull();
  if (skipLinkBounds === null) {
    throw new TypeError('Expected the focused skip link to be visible');
  }
  await skipLink.blur();

  // When the user opens the mobile navigation drawer.
  const toggle = page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL });
  const toggleBounds = await toggle.boundingBox();
  const themeBounds = await page
    .locator('header')
    .getByRole('button')
    .last()
    .boundingBox();
  await toggle.click();
  const drawer = page.getByRole('dialog', { name: 'Странична лента' });
  await expect
    .poll(async () => {
      const bounds = await drawer.boundingBox();
      return Math.abs(bounds?.y ?? Infinity);
    })
    .toBeLessThanOrEqual(ROUNDING_TOLERANCE);
  await expect
    .poll(async () => {
      const bounds = await drawer.boundingBox();
      return bounds?.height ?? -Infinity;
    })
    .toBeGreaterThanOrEqual(VIEWPORT_HEIGHT - ROUNDING_TOLERANCE);
  const drawerBounds = await drawer.boundingBox();
  const newChat = page.getByRole('button', { name: 'Нов разговор' });
  const newChatBounds = await newChat.boundingBox();
  const documentMetrics = await page.locator('html').evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));

  // Then the page shell, fixed drawer, and keyboard entry point clear the cutouts.
  expect(skipLinkBounds.y).toBeGreaterThanOrEqual(TOP_SAFE_AREA);
  expect(toggleBounds?.x).toBeGreaterThanOrEqual(LEFT_SAFE_AREA);
  expect(
    (themeBounds?.x ?? Infinity) + (themeBounds?.width ?? Infinity),
  ).toBeLessThanOrEqual(NARROW_VIEWPORT_WIDTH - RIGHT_SAFE_AREA);
  expect(newChatBounds?.x).toBeGreaterThanOrEqual(LEFT_SAFE_AREA);
  expect(Math.abs(drawerBounds?.y ?? Infinity)).toBeLessThanOrEqual(
    ROUNDING_TOLERANCE,
  );
  expect(drawerBounds?.height).toBeGreaterThanOrEqual(
    VIEWPORT_HEIGHT - ROUNDING_TOLERANCE,
  );
  expect(documentMetrics.scrollWidth).toBeLessThanOrEqual(
    documentMetrics.clientWidth,
  );

  // When a portaled settings dialog opens from the drawer.
  await drawerAccountMenu(page).click();
  await page.getByRole('menuitem', { name: 'API клучеви' }).click();
  const credentialsDialog = page.getByRole('dialog', {
    name: 'Лични API клучеви',
  });
  await expect(credentialsDialog).toBeVisible();
  await expect
    .poll(async () => {
      const bounds = await credentialsDialog.boundingBox();
      return bounds?.x ?? -Infinity;
    })
    .toBeGreaterThanOrEqual(LEFT_SAFE_AREA + SAFE_GUTTER);
  const dialogBounds = await credentialsDialog.boundingBox();
  expect(dialogBounds).not.toBeNull();
  if (dialogBounds === null) {
    throw new TypeError('Expected the credentials dialog to be visible');
  }

  // Then the dialog stays inside the safe vertical interval.
  expect(dialogBounds.x).toBeGreaterThanOrEqual(LEFT_SAFE_AREA + SAFE_GUTTER);
  expect(dialogBounds.x + dialogBounds.width).toBeLessThanOrEqual(
    NARROW_VIEWPORT_WIDTH - RIGHT_SAFE_AREA - SAFE_GUTTER + ROUNDING_TOLERANCE,
  );
  expect(dialogBounds.y).toBeGreaterThanOrEqual(TOP_SAFE_AREA + SAFE_GUTTER);
  expect(dialogBounds.y + dialogBounds.height).toBeLessThanOrEqual(
    VIEWPORT_HEIGHT - BOTTOM_SAFE_AREA - SAFE_GUTTER + ROUNDING_TOLERANCE,
  );
});

test('responsive dialog caps preserve safe gutters at the sm breakpoint', async ({
  page,
}) => {
  // Given a viewport wide enough to activate responsive dialog max-width classes.
  await prepareSafeAreaPage(page, WIDE_VIEWPORT_WIDTH);
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await drawerAccountMenu(page).click();

  // When the credential settings dialog opens.
  await page.getByRole('menuitem', { name: 'API клучеви' }).click();
  const credentialsDialog = page.getByRole('dialog', {
    name: 'Лични API клучеви',
  });
  await expect(credentialsDialog).toBeVisible();
  await expect
    .poll(async () => {
      const bounds = await credentialsDialog.boundingBox();
      return bounds?.x ?? -Infinity;
    })
    .toBeGreaterThanOrEqual(LEFT_SAFE_AREA + SAFE_GUTTER);
  const dialogBounds = await credentialsDialog.boundingBox();
  expect(dialogBounds).not.toBeNull();
  if (dialogBounds === null) {
    throw new TypeError('Expected the credentials dialog to be visible');
  }

  // Then responsive component caps cannot expand beyond the safe-area gutter.
  expect(dialogBounds.x).toBeGreaterThanOrEqual(LEFT_SAFE_AREA + SAFE_GUTTER);
  expect(dialogBounds.x + dialogBounds.width).toBeLessThanOrEqual(
    WIDE_VIEWPORT_WIDTH - RIGHT_SAFE_AREA - SAFE_GUTTER + ROUNDING_TOLERANCE,
  );
});

test('safe-area reduced height keeps credential errors and actions reachable', async ({
  page,
}) => {
  // Given a raw viewport above the compact breakpoint whose safe rectangle is short.
  await prepareSafeAreaPage(page, NARROW_VIEWPORT_WIDTH);
  await page.setViewportSize({ height: 560, width: NARROW_VIEWPORT_WIDTH });
  await page.route('**/api/chat/credentials', async (route) => {
    await route.fulfill({
      body: route.request().method() === 'PUT' ? '{}' : '[]',
      contentType: 'application/json',
      status: route.request().method() === 'PUT' ? 422 : 200,
    });
  });
  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await drawerAccountMenu(page).click();
  await page.getByRole('menuitem', { name: 'API клучеви' }).click();

  // When a rejected save adds a long error to the safe-area-bounded dialog.
  const dialog = page.getByRole('dialog', { name: 'Лични API клучеви' });
  await dialog.getByLabel('OpenAI API клуч').fill('sk-safe-area-error');
  await dialog
    .getByLabel('OpenAI Base URL (опционално)')
    .fill('https://blocked.example/v1');
  await dialog.getByRole('button', { name: 'Зачувај клучеви' }).click();
  await expect(dialog.getByRole('alert')).toBeVisible();

  // Then the provider body remains usable and the fixed actions clear the cutouts.
  const scroller = dialog.locator('form > div').first();
  const actionArea = dialog.locator('form > div').last();
  const dialogBounds = await dialog.boundingBox();
  const scrollerBounds = await scroller.boundingBox();
  const actionBounds = await actionArea.boundingBox();
  const saveBounds = await dialog
    .getByRole('button', { name: 'Зачувај клучеви' })
    .boundingBox();
  expect(scrollerBounds?.height).toBeGreaterThan(0);
  expect(
    (scrollerBounds?.y ?? Infinity) + (scrollerBounds?.height ?? Infinity),
  ).toBeLessThanOrEqual(actionBounds?.y ?? 0);
  expect(
    (saveBounds?.y ?? Infinity) + (saveBounds?.height ?? 0),
  ).toBeLessThanOrEqual(
    dialogBounds?.y === undefined ? 0 : dialogBounds.y + dialogBounds.height,
  );
  expect(await dialog.evaluate((element) => element.scrollTop)).toBe(0);
});

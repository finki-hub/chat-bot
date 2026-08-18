import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const LEFT_SAFE_AREA = 47;
const RIGHT_SAFE_AREA = 21;
const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';

test('landscape mobile controls stay outside display cutouts', async ({
  page,
}) => {
  // Given a landscape viewport with asymmetric safe-area insets.
  await page.setViewportSize({ height: 390, width: 844 });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setSafeAreaInsetsOverride', {
    insets: { left: LEFT_SAFE_AREA, right: RIGHT_SAFE_AREA },
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

  // When the user opens the mobile navigation drawer.
  const toggle = page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL });
  const toggleBounds = await toggle.boundingBox();
  const themeBounds = await page
    .locator('header')
    .getByRole('button')
    .last()
    .boundingBox();
  await toggle.click();
  const newChat = page.getByRole('button', { name: 'Нов разговор' });
  const newChatBounds = await newChat.boundingBox();
  const documentMetrics = await page.locator('html').evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));

  // Then both the page shell and fixed drawer clear the left cutout.
  expect(toggleBounds?.x).toBeGreaterThanOrEqual(LEFT_SAFE_AREA);
  expect(
    (themeBounds?.x ?? Infinity) + (themeBounds?.width ?? Infinity),
  ).toBeLessThanOrEqual(844 - RIGHT_SAFE_AREA);
  expect(newChatBounds?.x).toBeGreaterThanOrEqual(LEFT_SAFE_AREA);
  expect(documentMetrics.scrollWidth).toBeLessThanOrEqual(
    documentMetrics.clientWidth,
  );
});

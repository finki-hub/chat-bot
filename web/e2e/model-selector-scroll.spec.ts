import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

test('keeps model selector geometry stable at scroll boundaries', async ({
  page,
}) => {
  await mockModels(page);
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: '{}',
      contentType: 'application/json',
      status: 200,
    });
  });
  await installMockChatState(page, { streamUrl: 'http://127.0.0.1:9/chat' });

  await page.goto('/');
  await page.addStyleTag({
    content:
      'nextjs-portal { display: none !important; } [data-slot="select-content"] { animation: none !important; }',
  });

  await page.getByTestId('composer-model').click();

  const content = page.locator('[data-slot="select-content"]');
  const viewport = content.locator('[data-radix-select-viewport]');
  const scrollUpButton = content.locator(
    '[data-slot="select-scroll-up-button"]',
  );
  const scrollDownButton = content.locator(
    '[data-slot="select-scroll-down-button"]',
  );
  const scrollUpIndicator = content.locator(
    '[data-slot="select-scroll-up-indicator"]',
  );
  const scrollDownIndicator = content.locator(
    '[data-slot="select-scroll-down-indicator"]',
  );

  await viewport.evaluate((element) => {
    element.scrollTop = 0;
    element.dispatchEvent(new Event('scroll'));
  });
  await expect(scrollUpButton).toHaveCount(0);
  await expect(scrollDownButton).toBeVisible();
  await expect(scrollUpIndicator).toBeVisible();
  await expect(scrollDownIndicator).toBeVisible();
  const topBoundary = await content.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return { height: bounds.height, top: bounds.top };
  });

  await viewport.evaluate((element) => {
    element.scrollTop = (element.scrollHeight - element.clientHeight) / 2;
    element.dispatchEvent(new Event('scroll'));
  });
  await expect(scrollUpButton).toBeVisible();
  await expect(scrollDownButton).toBeVisible();
  const middle = await content.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return { height: bounds.height, top: bounds.top };
  });

  await viewport.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
    element.dispatchEvent(new Event('scroll'));
  });
  await expect(scrollUpButton).toBeVisible();
  await expect(scrollDownButton).toHaveCount(0);
  await expect(scrollUpIndicator).toBeVisible();
  await expect(scrollDownIndicator).toBeVisible();
  const bottomBoundary = await content.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return { height: bounds.height, top: bounds.top };
  });

  expect(middle.height).toBeCloseTo(topBoundary.height, 0);
  expect(middle.top).toBeCloseTo(topBoundary.top, 0);
  expect(bottomBoundary.height).toBeCloseTo(topBoundary.height, 0);
  expect(bottomBoundary.top).toBeCloseTo(topBoundary.top, 0);

  await page.keyboard.press('Escape');
  await page.setViewportSize({ height: 900, width: 1_280 });
  await page.getByTestId('composer-model').click();
  await expect(scrollUpIndicator).toBeHidden();
  await expect(scrollDownIndicator).toBeHidden();
});

import { expect, test } from '@playwright/test';

import type { ModelCatalog } from '@/lib/api-types';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const LONG_MODEL_NAME = 'Claude Sonnet 5 With A Deliberately Long Mobile Label';
const LONG_MODEL_CATALOG = {
  models: [
    {
      id: 'claude-sonnet-5',
      name: LONG_MODEL_NAME,
      provider: 'anthropic',
    },
  ],
  source: 'live',
  version: 1,
} as const satisfies ModelCatalog;

test('keeps a long model list inside narrow viewport gutters', async ({
  page,
}) => {
  // Given a narrow viewport and a model name wider than the available screen.
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
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page, {
    catalog: LONG_MODEL_CATALOG,
    credentialProviders: ['anthropic'],
  });
  await installMockChatState(page, { streamUrl: 'http://127.0.0.1:9/chat' });
  await page.goto('/');

  // When the model selector opens.
  await page.getByTestId('composer-model').click();
  const listbox = page.getByRole('listbox');
  await expect(listbox).toBeVisible();
  const box = await listbox.boundingBox();
  const option = page.getByRole('option', { name: LONG_MODEL_NAME });
  const chips = page.getByTestId('composer-chip-scroll');

  // Then the popup stays inside the configured 12px collision gutters.
  expect(box).not.toBeNull();
  if (box === null) {
    throw new TypeError('Expected the model listbox to have layout geometry');
  }
  expect(box.x).toBeGreaterThanOrEqual(12);
  expect(box.x + box.width).toBeLessThanOrEqual(308);
  await expect
    .poll(async () =>
      option.evaluate((element) => element.scrollWidth <= element.clientWidth),
    )
    .toBe(true);
  const chipMetrics = await chips.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(chipMetrics.scrollWidth).toBeLessThanOrEqual(chipMetrics.clientWidth);
});

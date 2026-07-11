import type { Page } from '@playwright/test';

import { CURATED_MODEL_DESCRIPTORS } from '@/lib/model-catalog';

export const MODEL_CATALOG = {
  models: Object.values(CURATED_MODEL_DESCRIPTORS),
  source: 'live',
  version: 1,
};

export const mockModels = async (page: Page): Promise<void> => {
  await page.route('**/api/models', async (route) => {
    await route.fulfill({
      body: JSON.stringify(MODEL_CATALOG),
      contentType: 'application/json',
      status: 200,
    });
  });
};

import type { Page } from '@playwright/test';

import { CURATED_MODEL_DESCRIPTORS } from '@/lib/model-catalog';

const MODEL_ORDER = [
  'gpt-5.4',
  'gpt-5.4-mini',
  'gpt-5.4-nano',
  'gemini-2.5-pro',
  'gemini-2.5-flash',
  'claude-opus-4-8',
  'claude-sonnet-5',
  'claude-haiku-4-5',
  'llama3.3:70b',
  'deepseek-r1:70b',
  'hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0',
  'hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0',
] as const;

export const MODEL_CATALOG = {
  models: MODEL_ORDER.map((id) => CURATED_MODEL_DESCRIPTORS[id]),
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

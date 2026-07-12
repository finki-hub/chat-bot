import type { Page } from '@playwright/test';

import type { ChatCredentialProvider } from '@/lib/api-types';

import { CURATED_MODEL_DESCRIPTORS } from '@/lib/model-catalog';

const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';
const USER_ID = '00000000-0000-4000-8000-000000000001';
const DEFAULT_CREDENTIAL_PROVIDERS = [
  'anthropic',
  'google',
  'ollama',
  'openai',
] as const satisfies readonly ChatCredentialProvider[];

export const MODEL_CATALOG = {
  models: Object.values(CURATED_MODEL_DESCRIPTORS),
  source: 'live',
  version: 1,
};

export const mockModels = async (
  page: Page,
  credentialProviders:
    | null
    | readonly ChatCredentialProvider[] = DEFAULT_CREDENTIAL_PROVIDERS,
): Promise<void> => {
  await page.route('**/api/models', async (route) => {
    await route.fulfill({
      body: JSON.stringify(MODEL_CATALOG),
      contentType: 'application/json',
      status: 200,
    });
  });
  if (credentialProviders === null) {
    return;
  }
  await page.route('**/api/chat/credentials', async (route) => {
    await route.fulfill({
      body: JSON.stringify(
        credentialProviders.map((provider) => ({
          [BASE_URL_FIELD]: null,
          [HAS_API_KEY_FIELD]: true,
          provider,
          [USER_ID_FIELD]: USER_ID,
        })),
      ),
      contentType: 'application/json',
      status: 200,
    });
  });
};

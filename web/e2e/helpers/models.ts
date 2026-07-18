import type { Page } from '@playwright/test';

import type { ChatCredentialProvider, ModelCatalog } from '@/lib/api-types';

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

type MockModelsInput =
  | MockModelsOptions
  | null
  | readonly ChatCredentialProvider[];

type MockModelsOptions = {
  readonly catalog?: (() => ModelCatalog) | ModelCatalog;
  readonly credentialProviders?:
    | (() => readonly ChatCredentialProvider[])
    | null
    | readonly ChatCredentialProvider[];
  readonly onModelsRequest?: () => void;
};

const isMockModelsOptions = (
  input: MockModelsInput,
): input is MockModelsOptions => input !== null && !Array.isArray(input);

export const mockModels = async (
  page: Page,
  input: MockModelsInput = DEFAULT_CREDENTIAL_PROVIDERS,
): Promise<void> => {
  const catalog = isMockModelsOptions(input)
    ? (input.catalog ?? MODEL_CATALOG)
    : MODEL_CATALOG;
  const credentialProviders = isMockModelsOptions(input)
    ? (input.credentialProviders ?? DEFAULT_CREDENTIAL_PROVIDERS)
    : input;
  const onModelsRequest = isMockModelsOptions(input)
    ? input.onModelsRequest
    : undefined;

  await page.route('**/api/models', async (route) => {
    onModelsRequest?.();
    await route.fulfill({
      body: JSON.stringify(typeof catalog === 'function' ? catalog() : catalog),
      contentType: 'application/json',
      status: 200,
    });
  });
  if (credentialProviders === null) {
    return;
  }
  await page.route('**/api/chat/credentials', async (route) => {
    const providers =
      typeof credentialProviders === 'function'
        ? credentialProviders()
        : credentialProviders;
    await route.fulfill({
      body: JSON.stringify(
        providers.map((provider) => ({
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

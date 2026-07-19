import { expect, type Page, test } from '@playwright/test';

import type { ModelCatalog } from '@/lib/api-types';

/* eslint-disable camelcase -- fixtures mirror the API wire contract. */
/* eslint-disable sonarjs/no-duplicate-string -- repeated selectors define the E2E surface. */
import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer } from './helpers/sse';

const STORAGE_KEY = 'finkiHub.ui';
// A model id that is no longer served by the catalog, to exercise stale recovery.
const STALE_MODEL = 'gpt-4o-mini-removed';
const USER_ID = '00000000-0000-4000-8000-000000000001';
const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';
const ACCOUNT_MENU_LABEL = /Корисничко мени:/u;
const LUNA_ID = 'gpt-5.6-luna';
const LUNA_NAME = 'GPT-5.6 Luna';
const EMPTY_JSON = '{}';
const LUNA_OPTION_PATTERN = /GPT-5\.6 Luna/u;
const lunaCatalog = (
  remaining: number,
  availability: 'both' | 'byok' | 'sponsored' = 'sponsored',
): ModelCatalog => ({
  models: [
    {
      availability,
      id: LUNA_ID,
      name: LUNA_NAME,
      provider: 'openai',
      sponsored_quota: {
        limit: 5,
        remaining,
        resets_at: '2099-01-01T12:00:00Z',
      },
    },
  ],
  source: 'live',
  version: 1,
});

const mockAuthenticatedSession = async (page: Page): Promise<void> => {
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
};

const mockSessionFor = async (
  page: Page,
  providerSubject: string,
): Promise<void> => {
  await page.route('**/api/auth/session', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        expires: '2099-01-01T00:00:00.000Z',
        user: {
          email: `${providerSubject}@example.com`,
          name: providerSubject,
          provider: 'github',
          providerSubject,
        },
      }),
      contentType: 'application/json',
      status: 200,
    });
  });
};

const hideDevelopmentOverlay = async (
  page: Parameters<typeof mockModels>[0],
) => {
  await page.addStyleTag({
    content: 'nextjs-portal { display: none !important; }',
  });
};

test.describe('model catalog selector (typed, mocked BFF)', () => {
  test('renders provider groups and recovers a stale selection', async ({
    page,
  }) => {
    await mockModels(page);
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 200,
      });
    });
    await installMockChatState(page, { streamUrl: 'http://127.0.0.1:9/chat' });

    // Persist a removed model id before the app boots so the recovery effect runs.
    await page.addInitScript(
      ({ key, model }: { key: string; model: string }) => {
        localStorage.setItem(
          key,
          JSON.stringify({
            state: { activeConversationId: null, model, reasoning: false },
            version: 0,
          }),
        );
      },
      { key: STORAGE_KEY, model: STALE_MODEL },
    );

    await page.goto('/');
    await hideDevelopmentOverlay(page);

    // Stale recovery: the trigger shows a valid catalog model (the recovered
    // default), not the placeholder it would show for the removed id.
    const trigger = page.getByTestId('composer-model');
    await expect(trigger).toContainText('Claude Sonnet 5');

    await trigger.click();

    await expect(page.getByTestId('model-provider-label')).toHaveText([
      'OpenAI',
      'Google / Gemini',
      'Anthropic',
      'Ollama',
    ]);

    // Model display names render (not raw ids).
    await expect(
      page.getByRole('option', { exact: true, name: 'GPT-5.4' }),
    ).toBeVisible();
    await expect(
      page.getByRole('option', { name: 'Qwen3 30B Thinking' }),
    ).toBeVisible();
    await page.screenshot({ path: 'test-results/model-selector-open-qa.png' });

    const selectedModel = page.getByRole('option', {
      exact: true,
      name: 'GPT-5.4 Mini',
    });
    await selectedModel.focus();
    await page.keyboard.press('Enter');
    await expect(trigger).toContainText('GPT-5.4 Mini');

    await page.setViewportSize({ height: 844, width: 390 });
    await trigger.click();
    await expect(page.getByTestId('model-provider-label')).toHaveText([
      'OpenAI',
      'Google / Gemini',
      'Anthropic',
      'Ollama',
    ]);
    await expect(page.getByTestId('model-tier-label')).toHaveCount(0);
    await page.keyboard.press('Escape');

    await page.getByTestId('composer-input').fill('Провери модел');
    const chatRequest = page.waitForRequest((request) => {
      const url = new URL(request.url());
      return url.pathname === '/api/chat' && request.method() === 'POST';
    });
    await page.getByTestId('composer-submit').click();

    const capturedRequest = await chatRequest;
    expect(capturedRequest.postDataJSON()).toMatchObject({
      model: 'gpt-5.4-mini',
    });

    await page.screenshot({ path: 'test-results/model-selector-qa.png' });
  });

  test('shows models without credentials as unavailable', async ({ page }) => {
    await mockModels(page, ['anthropic']);
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 200,
      });
    });
    await installMockChatState(page, { streamUrl: 'http://127.0.0.1:9/chat' });

    await page.goto('/');
    await hideDevelopmentOverlay(page);
    const trigger = page.getByTestId('composer-model');
    await expect(trigger).toContainText('Claude Sonnet 5');
    await trigger.click();

    const unavailable = page
      .getByRole('option')
      .filter({ hasText: 'GPT-5.4 Mini' });
    await expect(unavailable).toHaveAttribute('aria-disabled', 'true');
    await expect(unavailable).toContainText('Потребен е API клуч');
    await unavailable.dispatchEvent('click');
    await expect(trigger).toContainText('Claude Sonnet 5');
    await page.screenshot({
      animations: 'disabled',
      path: 'test-results/model-selector-unavailable-desktop.png',
    });
    await page.keyboard.press('Escape');
    await page.setViewportSize({ height: 900, width: 768 });
    await trigger.click();
    await page.screenshot({
      animations: 'disabled',
      path: 'test-results/model-selector-unavailable-tablet.png',
    });
    await page.keyboard.press('Escape');
    await page.setViewportSize({ height: 812, width: 375 });
    await trigger.click();
    await page.screenshot({
      animations: 'disabled',
      path: 'test-results/model-selector-unavailable-mobile.png',
    });
  });

  test('surfaces credential load errors and retries from settings', async ({
    page,
  }) => {
    let credentialRequests = 0;
    let credentialsAvailable = false;
    await mockAuthenticatedSession(page);
    await mockModels(page, null);
    await page.route('**/api/chat/credentials', async (route) => {
      credentialRequests += 1;
      await route.fulfill(
        credentialsAvailable
          ? {
              body: JSON.stringify([
                {
                  [BASE_URL_FIELD]: null,
                  [HAS_API_KEY_FIELD]: true,
                  provider: 'anthropic',
                  [USER_ID_FIELD]: USER_ID,
                },
              ]),
              contentType: 'application/json',
              status: 200,
            }
          : { body: EMPTY_JSON, contentType: 'application/json', status: 503 },
      );
    });
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 200,
      });
    });
    await installMockChatState(page, { streamUrl: 'http://127.0.0.1:9/chat' });

    await page.goto('/');
    await hideDevelopmentOverlay(page);
    const trigger = page.getByTestId('composer-model');
    await expect(trigger).toBeDisabled();
    await expect(trigger).toContainText('API клучевите се недостапни');

    await page.getByRole('button', { name: ACCOUNT_MENU_LABEL }).click();
    await page.getByRole('menuitem', { name: 'API клучеви' }).click();
    await expect(
      page.getByText('Клучевите не можеа да се вчитаат.'),
    ).toBeVisible();
    await page.screenshot({
      animations: 'disabled',
      path: 'test-results/credential-error-desktop.png',
    });
    credentialsAvailable = true;
    await page.getByRole('button', { name: 'Обиди се повторно' }).click();

    await expect(page.getByLabel('OpenAI API key')).toBeVisible();
    await expect(
      page.getByText('Клучевите не можеа да се вчитаат.'),
    ).toBeHidden();
    expect(credentialRequests).toBeGreaterThan(1);
  });

  test('renders the sponsored badge and updates the quota from five to zero', async ({
    page,
  }, testInfo) => {
    let catalog = lunaCatalog(5);
    await mockModels(page, {
      catalog: () => catalog,
      credentialProviders: [],
    });
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 200,
      });
    });
    const chatServer = await startChatStreamServer({
      gapMs: 0,
      head: [
        { messageMetadata: { inferenceModel: LUNA_ID }, type: 'start' },
        { id: 'sponsored-answer', type: 'text-start' },
        {
          delta: 'Квотата е ажурирана.',
          id: 'sponsored-answer',
          type: 'text-delta',
        },
        { id: 'sponsored-answer', type: 'text-end' },
        { type: 'finish' },
      ],
      tail: [],
    });
    try {
      await installMockChatState(page, {
        onCreate: () => {
          catalog = lunaCatalog(0);
        },
        streamUrl: chatServer.url,
      });

      await page.goto('/');
      const trigger = page.getByTestId('composer-model');
      await expect(trigger).toContainText(LUNA_NAME);
      await trigger.click();

      const badge = page.getByTestId(`model-free-badge-${LUNA_ID}`);
      await expect(badge).toContainText('Бесплатно');
      await expect(badge).toContainText('5/5');
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath('sponsored-quota-five.png'),
      });
      await page.keyboard.press('Escape');

      await page.getByTestId('composer-input').fill('Провери квота');
      await page.getByTestId('composer-submit').click();
      await expect
        .poll(() => catalog.models[0]?.sponsored_quota?.remaining)
        .toBe(0);

      await trigger.click();
      await expect(badge).toContainText('0/5');
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath('sponsored-quota-zero.png'),
      });
    } finally {
      await chatServer.close();
    }
  });

  test('refreshes sponsored availability after credentials change without a reload', async ({
    page,
  }) => {
    let catalog = lunaCatalog(5, 'byok');
    let credentialProviders: readonly ['openai'] | readonly [] = [];
    let modelRequests = 0;
    await mockSessionFor(page, 'credential-refresh');
    await mockModels(page, {
      catalog: () => catalog,
      credentialProviders: () => credentialProviders,
      onModelsRequest: () => {
        modelRequests += 1;
      },
    });
    await page.route('**/api/chat/credentials', async (route) => {
      if (route.request().method() === 'PUT') {
        catalog = lunaCatalog(5, 'both');
        credentialProviders = ['openai'];
        await route.fulfill({
          body: JSON.stringify({
            base_url: null,
            has_api_key: true,
            provider: 'openai',
            user_id: 'credential-refresh',
          }),
          contentType: 'application/json',
          status: 200,
        });
        return;
      }
      await route.fallback();
    });
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 200,
      });
    });
    await installMockChatState(page, {
      streamUrl: 'http://127.0.0.1:9/credentials',
    });

    await page.goto('/');
    await expect(page.getByTestId('composer-model')).toContainText('Модел');
    await page.getByTestId('composer-model').click();
    const unavailable = page.getByRole('option', { name: LUNA_OPTION_PATTERN });
    await expect(unavailable).toHaveAttribute('aria-disabled', 'true');
    await page.keyboard.press('Escape');

    await page.getByRole('button', { name: ACCOUNT_MENU_LABEL }).click();
    await page.getByRole('menuitem', { name: 'API клучеви' }).click();
    await page.getByLabel('OpenAI API key').fill('sk-test-refresh');
    await page.getByRole('button', { name: 'Зачувај' }).first().click();

    await expect.poll(() => modelRequests).toBeGreaterThan(1);
    await expect(
      page.getByRole('dialog', { name: 'Лични API клучеви' }),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Откажи' }).click();
    await page.getByTestId('composer-model').click();
    await expect(page.getByTestId(`model-free-badge-${LUNA_ID}`)).toContainText(
      '5/5',
    );
  });

  test('does not reuse sponsored quota between authenticated session subjects', async ({
    page,
  }) => {
    let sessionUser = 'session-a';
    let catalog = lunaCatalog(5);
    let modelRequests = 0;
    await page.route('**/api/auth/session', async (route) => {
      await route.fulfill({
        body: JSON.stringify({
          expires: '2099-01-01T00:00:00.000Z',
          user: {
            email: `${sessionUser}@example.com`,
            name: sessionUser,
            provider: 'github',
            providerSubject: sessionUser,
          },
        }),
        contentType: 'application/json',
        status: 200,
      });
    });
    await mockModels(page, {
      catalog: () => catalog,
      credentialProviders: [],
      onModelsRequest: () => {
        modelRequests += 1;
        if (sessionUser === 'session-b') {
          catalog = lunaCatalog(4);
        }
      },
    });
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 200,
      });
    });
    await installMockChatState(page, {
      streamUrl: 'http://127.0.0.1:9/session',
    });

    await page.goto('/');
    await page.getByTestId('composer-model').click();
    await expect(page.getByTestId(`model-free-badge-${LUNA_ID}`)).toContainText(
      '5/5',
    );
    await page.keyboard.press('Escape');

    sessionUser = 'session-b';
    await page.reload();
    await expect.poll(() => modelRequests).toBeGreaterThan(1);
    await page.getByTestId('composer-model').click();
    await expect(page.getByTestId(`model-free-badge-${LUNA_ID}`)).toContainText(
      '4/5',
    );
  });
});

/* eslint-enable camelcase -- end catalog wire fixtures. */
/* eslint-enable sonarjs/no-duplicate-string -- end repeated E2E selectors. */

import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const STORAGE_KEY = 'finkiHub.ui';
// A model id that is no longer served by the catalog, to exercise stale recovery.
const STALE_MODEL = 'gpt-4o-mini-removed';
const USER_ID = '00000000-0000-4000-8000-000000000001';
const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';

const hideDevelopmentOverlay = async (
  page: Parameters<typeof mockModels>[0],
) => {
  await page.addStyleTag({
    content: 'nextjs-portal { display: none !important; }',
  });
};

test.describe('model catalog selector (typed, mocked BFF)', () => {
  test('renders tier-first provider subgroups and recovers a stale selection', async ({
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

    await expect(page.getByTestId('model-tier-label')).toHaveText([
      'Премиум',
      'Стандарден',
      'Економичен',
    ]);
    await expect(page.getByTestId('model-provider-label').first()).toHaveText(
      'OpenAI',
    );

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
        body: '{}',
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
          : { body: '{}', contentType: 'application/json', status: 503 },
      );
    });
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: '{}',
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

    await page.getByRole('button', { name: 'API клучеви' }).click();
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
});

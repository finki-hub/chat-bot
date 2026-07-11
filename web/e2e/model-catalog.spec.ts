import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

const STORAGE_KEY = 'finkiHub.ui';
// A model id that is no longer served by the catalog, to exercise stale recovery.
const STALE_MODEL = 'gpt-4o-mini-removed';

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
});

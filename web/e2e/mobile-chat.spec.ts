import { expect, type Page, test } from '@playwright/test';

import type { ModelCatalog } from '@/lib/api-types';

/* eslint-disable camelcase -- catalog fixtures mirror the API wire contract. */
import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';

type MockSessionUser = {
  readonly email?: string;
  readonly name?: string;
};

const DEFAULT_SESSION_USER = {
  email: 'student@example.com',
  name: 'Student',
} as const satisfies MockSessionUser;
const CONVERSATION_TITLE = 'Услови за запишување семестар';
const SIDEBAR_TOGGLE_LABEL = 'Прикажи/сокриј странична лента';
const SIDEBAR_DIALOG_LABEL = 'Странична лента';
const CREDENTIALS_LABEL = 'API клучеви';
const ACCOUNT_MENU_LABEL = 'Корисничко мени: Student, student@example.com';
const CREDENTIAL_DIALOG_LABEL = 'Лични API клучеви';
const SAVE_CREDENTIALS_LABEL = 'Зачувај клучеви';
const EMPTY_JSON = '{}';
const STREAM_URL = 'http://127.0.0.1:9/stream';
const SPONSORED_MODEL = 'gpt-5.6-luna';
const SPONSORED_CATALOG: ModelCatalog = {
  models: [
    {
      availability: 'sponsored',
      id: SPONSORED_MODEL,
      name: 'GPT-5.6 Luna',
      provider: 'openai',
      sponsored_quota: {
        limit: 5,
        remaining: 3,
        resets_at: '2099-01-01T12:00:00Z',
      },
    },
  ],
  source: 'live',
  version: 1,
};
const mockSession = async (
  page: Page,
  user: MockSessionUser = DEFAULT_SESSION_USER,
): Promise<void> => {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.route('**/api/auth/session', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        expires: '2099-01-01T00:00:00.000Z',
        user,
      }),
      contentType: 'application/json',
      status: 200,
    });
  });
};

test('mobile drawer traps focus, closes with Escape, and restores the trigger', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const trigger = page.getByRole('button', {
    name: SIDEBAR_TOGGLE_LABEL,
  });
  await trigger.click();

  const drawer = page.getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL });
  await expect(drawer).toBeVisible();
  await expect(
    drawer.getByRole('button', { name: 'Нов разговор' }),
  ).toBeFocused();

  await page.keyboard.press('Shift+Tab');
  await expect(drawer).toContainText('Нов разговор');

  await page.keyboard.press('Escape');
  await expect(drawer).toBeHidden();
  await expect(trigger).toBeFocused();
});

test('mobile primary controls meet the 44px touch target minimum', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const controls = [
    page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }),
    page.getByRole('button', { name: 'Промени тема' }),
    page.getByTestId('composer-submit'),
  ];

  for (const control of controls) {
    const box = await control.boundingBox();
    expect(box?.height).toBeGreaterThanOrEqual(44);
    expect(box?.width).toBeGreaterThanOrEqual(44);
  }

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const drawer = page.getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL });
  const accountTrigger = drawer.getByRole('button', {
    name: ACCOUNT_MENU_LABEL,
  });
  const triggerBox = await accountTrigger.boundingBox();

  expect(triggerBox?.height).toBeGreaterThanOrEqual(44);
  expect(triggerBox?.width).toBeGreaterThanOrEqual(44);
});

test('mobile conversation actions use one touch-friendly overflow menu', async ({
  page,
}) => {
  // Given a conversation in the compact sidebar.
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, {
    conversations: [
      { id: 'conversation-1', model: null, title: CONVERSATION_TITLE },
    ],
    streamUrl: STREAM_URL,
  });
  await page.goto('/');
  const sidebarToggle = page.getByRole('button', {
    name: SIDEBAR_TOGGLE_LABEL,
  });
  await sidebarToggle.click();
  const drawer = page.getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL });

  // When the row action trigger is opened.
  const actionsTrigger = drawer.getByRole('button', {
    name: `Дејства за разговорот: ${CONVERSATION_TITLE}`,
  });
  const triggerBox = await actionsTrigger.boundingBox();
  await actionsTrigger.click();

  // Then one 44px trigger exposes labeled menu actions without crowding the title.
  expect(triggerBox?.height).toBeGreaterThanOrEqual(44);
  expect(triggerBox?.width).toBeGreaterThanOrEqual(44);
  await expect(
    page.getByRole('menuitem', { name: 'Генерирај име' }),
  ).toBeVisible();
  await expect(
    page.getByRole('menuitem', { name: 'Преименувај' }),
  ).toBeVisible();
  await expect(page.getByRole('menuitem', { name: 'Избриши' })).toBeVisible();
});

test('mobile keeps account actions in the profile menu and opens credentials after closing the drawer', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const sidebarToggle = page.getByRole('button', {
    name: SIDEBAR_TOGGLE_LABEL,
  });
  await sidebarToggle.click();
  const drawer = page.getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL });
  const accountTrigger = drawer.getByRole('button', {
    name: ACCOUNT_MENU_LABEL,
  });

  await expect(accountTrigger).toContainText('Student');
  await expect(accountTrigger).toContainText('student@example.com');
  await accountTrigger.click();
  await expect(page.getByRole('menuitem', { name: 'Одјави се' })).toBeVisible();
  await page.getByRole('menuitem', { name: CREDENTIALS_LABEL }).click();

  await expect(drawer).toBeHidden();
  const credentialsDialog = page.getByRole('dialog', {
    name: 'Лични API клучеви',
  });
  await expect(credentialsDialog).toBeVisible();
  await expect(
    page.getByRole('button', { name: CREDENTIALS_LABEL }),
  ).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Одјави се' })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(sidebarToggle).toBeFocused();
});

test('credential actions follow their visual order while the provider list alone scrolls', async ({
  page,
}) => {
  await page.setViewportSize({ height: 560, width: 375 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await page
    .getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL })
    .getByRole('button', {
      name: ACCOUNT_MENU_LABEL,
    })
    .click();
  await page.getByRole('menuitem', { name: CREDENTIALS_LABEL }).click();

  const dialog = page.getByRole('dialog', { name: CREDENTIAL_DIALOG_LABEL });
  const cancel = dialog.getByRole('button', { name: 'Откажи' });
  const save = dialog.getByRole('button', { name: SAVE_CREDENTIALS_LABEL });
  await dialog.getByLabel('OpenAI API клуч').fill('sk-order-test');
  const cancelBox = await cancel.boundingBox();
  const saveBox = await save.boundingBox();
  expect(cancelBox?.y).toBeLessThan(saveBox?.y ?? 0);
  await cancel.focus();
  await page.keyboard.press('Tab');
  await expect(save).toBeFocused();

  const dialogBox = await dialog.boundingBox();
  expect(dialogBox?.y).toBeGreaterThanOrEqual(0);
  expect(
    (dialogBox?.y ?? 0) + (dialogBox?.height ?? Infinity),
  ).toBeLessThanOrEqual(560);
  const scroller = dialog.locator('form > div').first();
  await expect
    .poll(() =>
      scroller.evaluate(
        (element) => element.scrollHeight > element.clientHeight,
      ),
    )
    .toBe(true);
  const footerBox = await dialog
    .locator('[data-slot="dialog-footer"]')
    .boundingBox();
  const footerY = footerBox?.y;
  await scroller.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  const scrolledFooterBox = await dialog
    .locator('[data-slot="dialog-footer"]')
    .boundingBox();
  const scrolledFooterY = scrolledFooterBox?.y;
  expect(Math.abs((scrolledFooterY ?? Infinity) - (footerY ?? 0))).toBeLessThan(
    1,
  );
});

test('credential save errors stay attributed and visible at responsive heights', async ({
  page,
}) => {
  await page.setViewportSize({ height: 640, width: 375 });
  await mockSession(page);
  await mockModels(page, []);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.route('**/api/chat/credentials', async (route) => {
    if (route.request().method() === 'PUT') {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 422,
      });
      return;
    }
    await route.fallback();
  });
  await page.goto('/');
  await page.addStyleTag({
    content: 'nextjs-portal { display: none !important; }',
  });

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await page
    .getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL })
    .getByRole('button', {
      name: ACCOUNT_MENU_LABEL,
    })
    .click();
  await page.getByRole('menuitem', { name: CREDENTIALS_LABEL }).click();

  const dialog = page.getByRole('dialog', { name: CREDENTIAL_DIALOG_LABEL });
  const openaiBaseUrl = dialog.getByLabel('OpenAI Base URL (опционално)');
  await dialog.getByLabel('OpenAI API клуч').fill('sk-visual-test');
  await openaiBaseUrl.fill('https://blocked.example/v1');
  await dialog.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }).click();

  await expect(openaiBaseUrl).toHaveAttribute('aria-invalid', 'true');
  await expect(dialog.getByRole('alert')).toContainText('OpenAI');
  await expect(dialog.getByRole('alert')).toContainText(
    'Base URL адресата не е дозволена.',
  );

  for (const viewport of [
    { height: 640, label: 'mobile-constrained-640', width: 375 },
    { height: 900, label: 'tablet', width: 768 },
    { height: 800, label: 'desktop', width: 1_280 },
  ] as const) {
    await page.setViewportSize(viewport);
    const alert = dialog.getByRole('alert');
    await expect(alert).toBeVisible();
    await expect(alert).toHaveCSS('text-wrap', 'pretty');
    await expect(dialog).toHaveCSS('overflow', 'clip');
    expect(
      await alert.evaluate((element) =>
        element.textContent.includes('за\u{00A0}стандардниот\u{00A0}endpoint'),
      ),
    ).toBe(true);
    expect(
      await dialog.evaluate((element) => {
        element.scrollTop = 100;
        return element.scrollTop;
      }),
    ).toBe(0);
    if (viewport.label === 'mobile-constrained-640') {
      await test.step('keeps the scroll body separate from fixed actions', async () => {
        const dialogBox = await dialog.boundingBox();
        const alertBox = await alert.boundingBox();
        const scroller = dialog.locator('form > div').first();
        const actionArea = dialog.locator('form > div').last();
        const scrollerBox = await scroller.boundingBox();
        const actionAreaBox = await actionArea.boundingBox();
        const saveBox = await dialog
          .getByRole('button', { name: SAVE_CREDENTIALS_LABEL })
          .boundingBox();
        expect(dialogBox?.y).toBeGreaterThanOrEqual(0);
        expect(
          (dialogBox?.y ?? 0) + (dialogBox?.height ?? Infinity),
        ).toBeLessThanOrEqual(viewport.height);
        expect(
          (scrollerBox?.y ?? Infinity) + (scrollerBox?.height ?? Infinity),
        ).toBeLessThanOrEqual(actionAreaBox?.y ?? 0);
        expect(alertBox?.y).toBeGreaterThanOrEqual(actionAreaBox?.y ?? 0);
        expect(alertBox?.y).toBeGreaterThanOrEqual(dialogBox?.y ?? 0);
        expect(scrollerBox?.height).toBeGreaterThan(0);
        expect(
          (alertBox?.y ?? Infinity) + (alertBox?.height ?? 0),
        ).toBeLessThan(saveBox?.y ?? 0);
        expect(
          (saveBox?.y ?? Infinity) + (saveBox?.height ?? 0),
        ).toBeLessThanOrEqual(viewport.height);
      });
    }
  }
});

test('all provider failures keep credential actions inside a constrained viewport', async ({
  page,
}) => {
  await page.setViewportSize({ height: 560, width: 375 });
  await mockSession(page);
  await mockModels(page, []);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.route('**/api/chat/credentials', async (route) => {
    if (route.request().method() === 'PUT') {
      await route.fulfill({
        body: EMPTY_JSON,
        contentType: 'application/json',
        status: 422,
      });
      return;
    }
    await route.fallback();
  });
  await page.goto('/');

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  await page
    .getByRole('dialog', { name: SIDEBAR_DIALOG_LABEL })
    .getByRole('button', { name: ACCOUNT_MENU_LABEL })
    .click();
  await page.getByRole('menuitem', { name: CREDENTIALS_LABEL }).click();

  const dialog = page.getByRole('dialog', { name: CREDENTIAL_DIALOG_LABEL });
  for (const provider of [
    'OpenAI',
    'Google / Gemini',
    'Anthropic',
    'Ollama',
  ] as const) {
    await dialog.getByLabel(`${provider} API клуч`).fill('sk-failure-test');
    await dialog
      .getByLabel(`${provider} Base URL (опционално)`)
      .fill('https://blocked.example/v1');
  }
  await dialog.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }).click();

  const alert = dialog.getByRole('alert');
  const save = dialog.getByRole('button', { name: SAVE_CREDENTIALS_LABEL });
  await expect(alert).toBeVisible();
  await expect(alert).toContainText('OpenAI');
  await expect(alert).toContainText('Ollama');
  const dialogBox = await dialog.boundingBox();
  const alertBox = await alert.boundingBox();
  const saveBox = await save.boundingBox();

  expect(alertBox?.y).toBeGreaterThanOrEqual(dialogBox?.y ?? 0);
  expect((saveBox?.y ?? Infinity) + (saveBox?.height ?? 0)).toBeLessThanOrEqual(
    560,
  );
});

test('credential dialog falls back to the persistent header trigger after a desktop-to-mobile resize', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ height: 800, width: 1_280 });
  await mockSession(page);
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');
  await page.addStyleTag({
    content: 'nextjs-portal { display: none !important; }',
  });

  await page
    .getByRole('button', {
      name: ACCOUNT_MENU_LABEL,
    })
    .click();
  await page.getByRole('menuitem', { name: CREDENTIALS_LABEL }).click();
  const credentialsDialog = page.getByRole('dialog', {
    name: CREDENTIAL_DIALOG_LABEL,
  });
  await expect(credentialsDialog).toBeVisible();
  const title = credentialsDialog.getByRole('heading', {
    name: CREDENTIAL_DIALOG_LABEL,
  });
  await expect(title).toBeFocused();
  await expect(title).toHaveCSS('outline-style', 'none');
  await page.screenshot({
    animations: 'disabled',
    path: testInfo.outputPath('credential-modal-final.png'),
  });

  await page.setViewportSize({ height: 900, width: 768 });
  await page.screenshot({
    animations: 'disabled',
    path: testInfo.outputPath('credential-modal-final-tablet.png'),
  });
  await page.setViewportSize({ height: 812, width: 375 });
  const sidebarToggle = page.getByRole('button', {
    name: SIDEBAR_TOGGLE_LABEL,
  });
  await page.screenshot({
    animations: 'disabled',
    path: testInfo.outputPath('credential-modal-final-mobile.png'),
  });
  await page.keyboard.press('Escape');
  await expect(sidebarToggle).toBeFocused();
});

test('mobile drawer truncates long authenticated names without overflowing', async ({
  page,
}) => {
  const name = 'Student With An Exceptionally Long Display Name';
  await page.setViewportSize({ height: 812, width: 320 });
  await mockSession(page, { email: 'student@example.com', name });
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const identity = page.getByTestId('sidebar-user-identity');
  const label = identity.getByText(name);
  await expect(label).toBeVisible();
  await expect(label).toHaveCSS('overflow', 'hidden');
  await expect(label).toHaveCSS('text-overflow', 'ellipsis');
  await expect(label).toHaveCSS('white-space', 'nowrap');
  await expect
    .poll(async () =>
      label.evaluate((element) => element.scrollWidth > element.clientWidth),
    )
    .toBe(true);
  const box = await identity.boundingBox();
  expect((box?.x ?? Infinity) + (box?.width ?? Infinity)).toBeLessThanOrEqual(
    320,
  );
});

test('mobile keeps the account menu available when authenticated identity is unavailable', async ({
  page,
}) => {
  await page.setViewportSize({ height: 812, width: 320 });
  await mockSession(page, {});
  await mockModels(page);
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  await expect(page.getByRole('button', { name: 'Одјави се' })).toHaveCount(0);
  await expect(
    page.getByRole('button', { name: CREDENTIALS_LABEL }),
  ).toHaveCount(0);

  await page.getByRole('button', { name: SIDEBAR_TOGGLE_LABEL }).click();
  const accountTrigger = page.getByRole('button', {
    name: 'Корисничко мени: Сметка',
  });

  await expect(accountTrigger).toContainText('Сметка');
  await accountTrigger.click();
  await expect(
    page.getByRole('menuitem', { name: CREDENTIALS_LABEL }),
  ).toBeVisible();
  await expect(page.getByRole('menuitem', { name: 'Одјави се' })).toBeVisible();
});

test('mobile model selector exposes the sponsored badge and remaining quota', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ height: 812, width: 375 });
  await mockSession(page);
  await mockModels(page, {
    catalog: SPONSORED_CATALOG,
    credentialProviders: [],
  });
  await installMockChatState(page, { streamUrl: STREAM_URL });
  await page.goto('/');

  const trigger = page.getByTestId('composer-model');
  await expect(trigger).toContainText('GPT-5.6 Luna');
  await trigger.click();

  await expect(page.getByTestId('model-provider-label')).toHaveText('OpenAI');
  const badge = page.getByTestId(`model-free-badge-${SPONSORED_MODEL}`);
  await expect(badge).toContainText('Бесплатно');
  await expect(badge).toContainText('3/5');
  await page.screenshot({
    animations: 'disabled',
    path: testInfo.outputPath('sponsored-selector-mobile.png'),
  });
});

/* eslint-enable camelcase -- end catalog wire fixtures. */

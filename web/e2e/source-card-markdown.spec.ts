import { expect, test } from '@playwright/test';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer, type UiChunk } from './helpers/sse';

const ARIA_EXPANDED = 'aria-expanded';
const COMPOSER_INPUT = 'composer-input';
const COMPOSER_SUBMIT = 'composer-submit';
const INFERENCE_MODEL = 'claude-sonnet-5';
const NEXT_DEV_LAUNCHER_STYLE = 'nextjs-portal { display: none !important; }';
const RESPONSE_ID = '11111111-2222-3333-4444-555555555555';
const VIEWPORT_WIDTHS = [375, 768, 1_280] as const;

test('renders source Markdown in collapsed and expanded cards', async ({
  page,
}, testInfo) => {
  const answerId = 'txt-source-answer';
  const markdownText =
    'Клучните услови со доволно зборови за скратената верзија што останува јасна и продолжува понатаму';
  const requestedExternalUrls: string[] = [];
  const sourceTitle = 'Упис на семестар';
  const sources = [
    {
      id: 'q1',
      kind: 'faq',
      snippet: `**${markdownText}** со *проширена содржина*.\n\n[врска](https://example.com/)\n\n![tracking](https://example.com/pixel.png)\n\n\`\`\`mermaid\nflowchart LR\nA@{ img: "https://evil.example/pixel.png" }\nclick A "https://evil.example/"\n\`\`\`\n\n<picture><source srcset="https://evil.example/tracker.png"><img src="https://evil.example/fallback.png" alt="tracker"></picture>`,
      title: sourceTitle,
    },
    { id: 'q2', kind: 'faq', title: 'Заверка на семестар' },
  ] as const;
  const chatServer = await startChatStreamServer({
    gapMs: 3_000,
    head: [
      {
        messageMetadata: {
          inferenceModel: INFERENCE_MODEL,
          responseId: RESPONSE_ID,
        },
        type: 'start',
      },
      { id: answerId, type: 'text-start' },
      {
        delta: 'Одговорот се генерира и изворите се подготвени.',
        id: answerId,
        type: 'text-delta',
      },
      { messageMetadata: { sources }, type: 'message-metadata' },
    ] satisfies UiChunk[],
    tail: [{ id: answerId, type: 'text-end' }, { type: 'finish' }],
  });

  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ ok: true }),
      contentType: 'application/json',
      status: 200,
    });
  });
  await page.route('**/api/chat/*/stream', async (route) => {
    await route.fulfill({ status: 204 });
  });
  page.on('request', (request) => {
    if (request.url().includes('evil.example')) {
      requestedExternalUrls.push(request.url());
    }
  });
  await mockModels(page);
  await installMockChatState(page, { streamUrl: chatServer.url });
  await page.goto('/');
  await page.getByTestId(COMPOSER_INPUT).fill('Кои се условите за упис?');
  await page.getByTestId(COMPOSER_SUBMIT).click();

  const collapsedToggle = page.getByRole('button', {
    name: 'Прикажи извори',
  });
  await expect(collapsedToggle).toHaveAttribute(ARIA_EXPANDED, 'false');
  await expect(page.getByText(sourceTitle)).toHaveCount(0);

  await expect(
    page.getByRole('button', { name: 'Сокриј извори' }),
  ).toHaveAttribute(ARIA_EXPANDED, 'true', { timeout: 10_000 });
  await expect(page.getByText(sourceTitle)).toBeVisible();
  await expect(page.getByText(markdownText, { exact: true })).toBeVisible();
  await expect(page.getByText(`**${markdownText}**`)).toHaveCount(0);
  const sourceCard = page.getByRole('button', {
    name: new RegExp(sourceTitle, 'u'),
  });
  const sourceSnippetRegion = sourceCard.locator(
    'xpath=following-sibling::*[1]',
  );

  await expect(sourceCard).toHaveAttribute(ARIA_EXPANDED, 'false');
  await expect(sourceSnippetRegion).toHaveAttribute('aria-hidden', 'true');
  await expect(sourceSnippetRegion.locator('strong')).toHaveText(markdownText);
  await expect(
    sourceSnippetRegion.locator(
      'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
    ),
  ).toHaveCount(0);
  await expect(sourceSnippetRegion.locator('img')).toHaveCount(0);
  await expect(sourceSnippetRegion.locator('code')).toContainText(
    'flowchart LR',
  );
  await expect(
    sourceSnippetRegion.locator('svg image, picture, source'),
  ).toHaveCount(0);

  for (const width of VIEWPORT_WIDTHS) {
    await page.setViewportSize({ height: 900, width });
    await expect(sourceSnippetRegion).toBeVisible();
    await page.screenshot({
      animations: 'disabled',
      path: testInfo.outputPath(`source-card-collapsed-${String(width)}.png`),
      style: NEXT_DEV_LAUNCHER_STYLE,
    });
  }

  await sourceCard.focus();
  await page.keyboard.press('Space');
  await expect(sourceCard).toHaveAttribute(ARIA_EXPANDED, 'true');
  await expect(sourceSnippetRegion).not.toHaveAttribute('aria-hidden');
  await expect(sourceSnippetRegion.locator('em')).toHaveText(
    'проширена содржина',
  );
  const markdownLink = page.getByRole('button', { name: 'врска' });

  await expect(markdownLink).toBeVisible();
  await markdownLink.focus();
  await expect(markdownLink).toBeFocused();
  await markdownLink.click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(sourceCard).toHaveAttribute(ARIA_EXPANDED, 'true');

  for (const width of VIEWPORT_WIDTHS) {
    await page.setViewportSize({ height: 900, width });
    await expect(sourceSnippetRegion.locator('strong')).toBeVisible();
    await expect(sourceSnippetRegion.locator('em')).toBeVisible();
    await page.screenshot({
      animations: 'disabled',
      path: testInfo.outputPath(`source-card-expanded-${String(width)}.png`),
      style: NEXT_DEV_LAUNCHER_STYLE,
    });
  }

  expect(requestedExternalUrls).toStrictEqual([]);

  await chatServer.close();
});

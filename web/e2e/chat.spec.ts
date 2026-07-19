import { expect, test } from '@playwright/test';

import type { ModelCatalog } from '@/lib/api-types';
import type { ChatConversationHistory } from '@/lib/conversation-types';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import {
  stagedRunChunks,
  startChatStreamServer,
  toolRunChunks,
} from './helpers/sse';

const RESPONSE_ID = '11111111-2222-3333-4444-555555555555';
const INFERENCE_MODEL = 'claude-sonnet-5';
const PREAMBLE = 'Дозволете да проверам…';
const STATUS_LABEL = '🔍 Пребарувам…';
const TOOL = 'search_documents';
const ANSWER = 'Резултатите од испитите се објавуваат на https://finki.ukim.mk';
const LINK_NAME = /finki\.ukim\.mk/u;
const DIAGNOSTICS_LABEL = /Дијагностика/u;
const OBSERVABLE_STAGE_GAP_MS = 5_000;
const ANSWER_TEST_ID = 'answer-text';
const ANSWER_PREVIEW = 'Резултатите од испитите се објавуваат';
const SPONSORED_MODEL = 'gemini-3.5-flash';
const COMPOSER_INPUT = ['composer', 'input'].join('-');
const COMPOSER_SUBMIT = ['composer', 'submit'].join('-');
/* eslint-disable camelcase -- catalog fixture mirrors the API wire contract. */
const SPONSORED_CATALOG: ModelCatalog = {
  models: [
    {
      availability: 'sponsored',
      id: SPONSORED_MODEL,
      name: 'Gemini 3.5 Flash',
      provider: 'google',
      sponsored_quota: {
        limit: 5,
        remaining: 0,
        resets_at: '2099-01-01T12:00:00Z',
      },
    },
  ],
  source: 'live',
  version: 1,
};
/* eslint-enable camelcase -- end catalog wire fixture. */
const EVIDENCE_DIR = `${process.cwd()}/../.omo/evidence/ulw/ses_08b75c4cdffe8g6tX0apGLd65d/task-12-cross-surface-sponsored-luna`;

const errorStream = (code: string, message: string) =>
  startChatStreamServer({
    gapMs: 0,
    head: [
      { messageMetadata: { inferenceModel: SPONSORED_MODEL }, type: 'start' },
      {
        data: { code, message },
        transient: true,
        type: 'data-error',
      },
    ],
    tail: [],
  });

test.describe('chat streaming (mocked BFF)', () => {
  test('shows the search chip, drops the preamble, renders the answer, and toggles like feedback', async ({
    page,
  }) => {
    const chatChunks = toolRunChunks({
      answer: ANSWER,
      inferenceModel: INFERENCE_MODEL,
      preamble: PREAMBLE,
      responseId: RESPONSE_ID,
      statusLabel: STATUS_LABEL,
      tool: TOOL,
    });
    const diagnosticsChunk = {
      messageMetadata: {
        diagnostics: {
          cost: { inputUsd: 0.00003, outputUsd: 0.00045, totalUsd: 0.00048 },
          serverTotalMs: 1_700,
          serverTtftMs: 200,
          spans: {
            'links.rerank': 120,
            'retrieval.hyde': 180,
            'retrieval.query_rewrite': 160,
            'retrieval.query_transform': 350,
          },
          tokens: { input: 10, output: 30, total: 40 },
        },
        inferenceModel: INFERENCE_MODEL,
        responseId: RESPONSE_ID,
      },
      type: 'message-metadata',
    } satisfies (typeof chatChunks)[number];
    const chunks = [
      ...chatChunks.slice(0, -1),
      diagnosticsChunk,
      ...chatChunks.slice(-1),
    ];
    const statusIndex = chunks.findIndex((c) => c.type === 'data-status');
    const chatServer = await startChatStreamServer({
      gapMs: OBSERVABLE_STAGE_GAP_MS,
      head: chunks.slice(0, statusIndex + 1),
      tail: chunks.slice(statusIndex + 1),
    });

    await mockModels(page);

    await installMockChatState(page, { streamUrl: chatServer.url });

    const feedbackRequests: Array<{
      readonly body: Record<string, unknown>;
      readonly method: string;
    }> = [];
    await page.route('**/api/feedback', async (route) => {
      const request = route.request();
      feedbackRequests.push({
        body: request.postDataJSON() as Record<string, unknown>,
        method: request.method(),
      });
      await route.fulfill({
        body: JSON.stringify({
          /* eslint-disable camelcase -- snake_case mirrors the FeedbackAck wire contract */
          feedback_type: request.method() === 'DELETE' ? null : 'like',
          id: RESPONSE_ID,
          response_id: RESPONSE_ID,
          /* eslint-enable camelcase -- re-enable past the wire-shaped ack */
        }),
        contentType: 'application/json',
        status: 200,
      });
    });

    await page.goto('/');

    const input = page.getByTestId(COMPOSER_INPUT);
    await input.fill('Кога се објавуваат резултатите?');
    await page.getByTestId(COMPOSER_SUBMIT).click();

    const chip = page.getByTestId('search-status');
    await expect(chip).toBeVisible({ timeout: 15_000 });
    await expect(chip).toContainText('Пребарувам');

    const answer = page.getByTestId(ANSWER_TEST_ID);
    await expect(answer).toContainText(ANSWER_PREVIEW, { timeout: 15_000 });
    const autolink = answer.locator('[data-streamdown="link"]', {
      hasText: LINK_NAME,
    });
    await expect(autolink).toBeVisible();

    await expect(page.getByText(PREAMBLE)).toHaveCount(0);

    const timing = page.getByTestId('message-timing');
    await expect(timing).toBeVisible();
    await expect(timing).toContainText('прв токен');
    const diagnosticsTrigger = page.getByRole('button', {
      name: DIAGNOSTICS_LABEL,
    });
    await expect(diagnosticsTrigger).toBeVisible();
    await diagnosticsTrigger.focus();
    await expect(page.getByText('trace ID')).toBeVisible();
    await expect(page.getByText(RESPONSE_ID)).toBeVisible();
    await expect(page.getByText('рерангирање врски')).toBeVisible();
    await expect(page.getByText('хипотетички документ (HyDE)')).toBeVisible();
    await expect(page.getByText('преформулација на прашање')).toBeVisible();
    await expect(page.getByText('трансформација на прашање')).toBeVisible();
    await expect(page.getByText('links.rerank')).toHaveCount(0);
    await expect(page.getByText('цена')).toBeVisible();
    await expect(page.getByText('$0.000480')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByText('trace ID')).toHaveCount(0);

    const like = page.getByTestId('like-button');
    await like.click();
    await expect(like).toHaveAttribute('aria-pressed', 'true');
    await like.click();
    await expect(like).toHaveAttribute('aria-pressed', 'false');
    await expect.poll(() => feedbackRequests).toHaveLength(2);
    expect(feedbackRequests).toStrictEqual([
      {
        body: { feedbackType: 'like', responseId: RESPONSE_ID },
        method: 'POST',
      },
      {
        body: { responseId: RESPONSE_ID },
        method: 'DELETE',
      },
    ]);
    await expect(page.getByTestId('dislike-button')).toHaveAttribute(
      'aria-pressed',
      'false',
    );

    await chatServer.close();
  });

  test('streams the retrieval stepper through stages, then the answer', async ({
    page,
  }) => {
    const chunks = stagedRunChunks({
      answer: ANSWER,
      inferenceModel: INFERENCE_MODEL,
      responseId: RESPONSE_ID,
      statusLabel: STATUS_LABEL,
      tool: TOOL,
    });
    // Split so the stepper is observable mid-pipeline: head ends at the
    // `context` stage; tail delivers the reset + answer after a gap.
    const resetIndex = chunks.findIndex((c) => c.type === 'data-reset');
    const chatServer = await startChatStreamServer({
      gapMs: OBSERVABLE_STAGE_GAP_MS,
      head: chunks.slice(0, resetIndex),
      tail: chunks.slice(resetIndex),
    });

    await mockModels(page);

    await installMockChatState(page, { streamUrl: chatServer.url });

    await page.goto('/');

    const input = page.getByTestId('composer-input');
    await input.fill('Кога се објавуваат резултатите?');
    await page.getByTestId('composer-submit').click();

    const stepper = page.getByTestId('search-stepper');
    await expect(stepper).toBeVisible({ timeout: 15_000 });
    // Progressive reveal: retrieval stages appear one-by-one as they run.
    await expect(stepper).toContainText('Разбирање…');
    await expect(stepper).toContainText('Пребарување…');
    await expect(stepper).toContainText('Рерангирање…');
    await expect(stepper).toContainText('Составување…');

    const answer = page.getByTestId(ANSWER_TEST_ID);
    await expect(answer).toContainText(ANSWER_PREVIEW, { timeout: 15_000 });
    // Once the answer streams in, the stepper is replaced by the answer.
    await expect(stepper).toHaveCount(0);

    await chatServer.close();
  });

  test('shows quota exhaustion actions and preserves the prior conversation', async ({
    page,
  }) => {
    await page.setViewportSize({ height: 812, width: 375 });
    const conversationId = 'sponsored-preserved-conversation';
    const history = {
      conversation: {
        id: conversationId,
        model: SPONSORED_MODEL,
        title: 'Постоечки разговор',
      },
      messages: [
        {
          id: 'prior-assistant',
          metadata: { inferenceModel: SPONSORED_MODEL },
          parts: [{ text: 'Претходен одговор', type: 'text' }],
          role: 'assistant',
        },
      ],
    } satisfies ChatConversationHistory;
    const chatServer = await errorStream(
      'free_quota_exhausted',
      'backend quota detail must not render',
    );

    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: '{}',
        contentType: 'application/json',
        status: 200,
      });
    });
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
    await mockModels(page, {
      catalog: SPONSORED_CATALOG,
      credentialProviders: [],
    });
    await installMockChatState(page, {
      conversations: [
        {
          id: conversationId,
          model: SPONSORED_MODEL,
          title: 'Постоечки разговор',
        },
      ],
      histories: { [conversationId]: history },
      streamUrl: chatServer.url,
    });
    await page.goto('/');
    await page
      .getByRole('button', { name: 'Прикажи/сокриј странична лента' })
      .click();
    await page
      .getByRole('button', { exact: true, name: 'Постоечки разговор' })
      .click();
    await expect(page.getByText('Претходен одговор')).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByTestId('composer-model')).toContainText(
      'Gemini 3.5 Flash',
    );
    await expect(page.getByTestId('composer-input')).toBeEnabled();
    await page.getByTestId('composer-input').fill('Ново прашање');
    await page.getByTestId('composer-submit').click();

    await expect(
      page.getByText('Бесплатната квота е искористена.'),
    ).toBeVisible();
    await expect(
      page.getByText('backend quota detail must not render'),
    ).toHaveCount(0);
    await expect(
      page.getByRole('button', { name: 'Додај API клуч' }),
    ).toBeVisible();
    await expect(
      page.getByRole('button', { name: 'Почекај до ресетирањето' }),
    ).toBeVisible();
    await expect(page.getByText('Претходен одговор')).toBeVisible();
    await page.screenshot({
      animations: 'disabled',
      path: `${EVIDENCE_DIR}/sponsored-exhaustion-preserved-conversation.png`,
    });

    await page.getByRole('button', { name: 'Почекај до ресетирањето' }).click();
    await expect(page.getByText('Претходен одговор')).toBeVisible();
    await chatServer.close();
  });

  test('renders global sponsored unavailability without exposing backend details', async ({
    page,
  }) => {
    const chatServer = await errorStream(
      'free_tier_unavailable',
      'upstream availability detail must not render',
    );
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        body: '{}',
        contentType: 'application/json',
        status: 200,
      });
    });
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
    await mockModels(page, {
      catalog: SPONSORED_CATALOG,
      credentialProviders: [],
    });
    await installMockChatState(page, { streamUrl: chatServer.url });

    await page.goto('/');
    await page.getByTestId('composer-input').fill('Провери достапност');
    await page.getByTestId('composer-submit').click();

    await expect(
      page.getByText('Бесплатниот модел е привремено недостапен.'),
    ).toBeVisible();
    await expect(
      page.getByText('upstream availability detail must not render'),
    ).toHaveCount(0);
    await expect(
      page.getByRole('button', { name: 'Додај API клуч' }),
    ).toHaveCount(0);
    await expect(
      page.getByRole('button', { name: 'Почекај до ресетирањето' }),
    ).toHaveCount(0);
    await chatServer.close();
  });
});

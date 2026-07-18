import { expect, type Page, test } from '@playwright/test';

import type { ModelCatalog } from '@/lib/api-types';

/* eslint-disable camelcase -- catalog fixtures mirror the API wire contract. */
import { mockModels } from './helpers/models';
import { startChatStreamServer, type UiChunk } from './helpers/sse';

const RESPONSE_ID = '22222222-3333-4444-5555-666666666666';
const INFERENCE_MODEL = 'claude-sonnet-5';
const FIRST_TOKEN = 'Прв дел од одговорот';
const FINAL_TOKEN = ' и продолжение по освежување.';
const STOP_TOKEN = 'Делумен одговор';
const LONG_GAP_MS = 30_000;
const CHAT_STREAM_URL_PATTERN = /\/api\/chat\/[^/]+\/stream$/u;
const EVIDENCE_DIR = `${process.cwd()}/../.omo/evidence/ulw/ses_08b75c4cdffe8g6tX0apGLd65d/task-12-cross-surface-sponsored-luna`;
const SPONSORED_CATALOG: ModelCatalog = {
  models: [
    {
      availability: 'sponsored',
      id: INFERENCE_MODEL,
      name: 'Claude Sonnet 5',
      provider: 'anthropic',
      sponsored_quota: {
        limit: 5,
        remaining: 5,
        resets_at: '2099-01-01T12:00:00Z',
      },
    },
  ],
  source: 'live',
  version: 1,
};

type ConversationRow = {
  readonly id: string;
  readonly model: string;
  readonly title: string;
};

type PostRequestBody = {
  readonly id?: unknown;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseConversationId = (body: null | string): string => {
  if (body === null) {
    throw new Error('chat request did not include a body');
  }

  const parsed: unknown = JSON.parse(body);
  const requestBody: PostRequestBody = isRecord(parsed) ? parsed : {};

  if (typeof requestBody.id !== 'string' || requestBody.id.length === 0) {
    throw new Error('chat request did not include a conversation id');
  }

  return requestBody.id;
};

const conversationIdFrom = (routeUrl: string): string =>
  decodeURIComponent(
    new URL(routeUrl).pathname.split('/', 4)[3] ?? 'conversation',
  );

const emptyHistoryBody = (id: null | string): string =>
  JSON.stringify({
    conversation: {
      id: id ?? 'conversation',
      model: INFERENCE_MODEL,
      title: 'New conversation',
    },
    messages: [],
  });

const chunksForAnswer = (answer: string): UiChunk[] => [
  {
    messageMetadata: {
      inferenceModel: INFERENCE_MODEL,
      responseId: RESPONSE_ID,
    },
    type: 'start',
  },
  { id: 'txt-answer', type: 'text-start' },
  { delta: answer, id: 'txt-answer', type: 'text-delta' },
  { id: 'txt-answer', type: 'text-end' },
  { type: 'finish' },
];

const installModelRoute = async (
  page: Page,
  catalog: ModelCatalog = SPONSORED_CATALOG,
): Promise<void> => {
  await mockModels(page, { catalog });
};

const installHealthRoute = async (page: Page): Promise<void> => {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({ status: 'ok' }),
      contentType: 'application/json',
      status: 200,
    });
  });
};

test.describe('resumable chat lifecycle (mocked BFF)', () => {
  test('resumes after refresh and persists the completed assistant response', async ({
    page,
  }) => {
    // Given: the initial POST emits one token slowly while the resume endpoint can replay the full stream.
    await installHealthRoute(page);
    await installModelRoute(page);
    const firstLegServer = await startChatStreamServer({
      gapMs: LONG_GAP_MS,
      head: chunksForAnswer(FIRST_TOKEN).slice(0, 3),
      tail: [],
    });
    let conversationId: null | string = null;
    let historyRequests = 0;
    let resumeRequests = 0;
    let allowResumeReplay = false;
    let allowServerHistory = false;
    const conversations: ConversationRow[] = [];

    await page.route('**/api/chat', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          body: JSON.stringify(conversations),
          contentType: 'application/json',
          status: 200,
        });
        return;
      }
      conversationId = parseConversationId(route.request().postData());
      await route.fulfill({
        headers: { location: firstLegServer.url },
        status: 307,
      });
    });
    await page.route('**/api/chat/*', async (route) => {
      if (route.request().method() === 'PATCH') {
        conversationId = conversationIdFrom(route.request().url());
        if (
          conversations.every(
            (conversation) => conversation.id !== conversationId,
          )
        ) {
          conversations.unshift({
            id: conversationId,
            model: INFERENCE_MODEL,
            title: 'Резимирај ми го условот за запишување.',
          });
        }
        await route.fulfill({ status: 204 });
        return;
      }
      await route.fallback();
    });
    await page.route('**/api/chat/*/stream', async (route) => {
      resumeRequests += 1;
      if (!allowResumeReplay) {
        await route.fulfill({ status: 204 });
        return;
      }
      allowResumeReplay = false;
      const body = chunksForAnswer(`${FIRST_TOKEN}${FINAL_TOKEN}`)
        .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
        .join('');
      await route.fulfill({
        body: `${body}data: [DONE]\n\n`,
        contentType: 'text/event-stream',
        headers: { 'x-vercel-ai-ui-message-stream': 'v1' },
        status: 200,
      });
    });
    await page.route('**/api/chat/*/history', async (route) => {
      historyRequests += 1;
      if (!allowServerHistory) {
        await route.fulfill({
          body: emptyHistoryBody(conversationId),
          contentType: 'application/json',
          status: 200,
        });
        return;
      }
      await route.fulfill({
        body: JSON.stringify({
          conversation: {
            id: conversationId,
            model: INFERENCE_MODEL,
            title: 'Резимирај ми го условот за запишување.',
          },
          messages: [
            {
              id: 'u-history',
              metadata: {},
              parts: [
                {
                  text: 'Резимирај ми го условот за запишување.',
                  type: 'text',
                },
              ],
              role: 'user',
            },
            {
              id: 'a-history',
              metadata: {
                inferenceModel: INFERENCE_MODEL,
                responseId: RESPONSE_ID,
              },
              parts: [{ text: `${FIRST_TOKEN}${FINAL_TOKEN}`, type: 'text' }],
              role: 'assistant',
            },
          ],
        }),
        contentType: 'application/json',
        status: 200,
      });
    });

    await page.goto('/');
    await page.getByTestId('composer-model').click();
    await expect(
      page.getByTestId('model-free-badge-claude-sonnet-5'),
    ).toContainText('5/5');
    await page.keyboard.press('Escape');
    await page
      .getByTestId('composer-input')
      .fill('Резимирај ми го условот за запишување.');
    await page.getByTestId('composer-input').press('Enter');
    await expect(page.getByTestId('answer-text')).toContainText(FIRST_TOKEN);

    // When: the browser refreshes mid-generation and reconnects to the same chat.
    allowResumeReplay = true;
    const resumeResponse = page.waitForResponse(
      (response) =>
        CHAT_STREAM_URL_PATTERN.test(response.url()) &&
        response.status() === 200,
    );
    await page.reload();

    // Then: the UI consumes the resumed SSE and can reload the final answer from server history.
    const resumed = await resumeResponse;
    expect(resumed.status()).toBe(200);
    await expect(page.getByTestId('answer-text').last()).toContainText(
      `${FIRST_TOKEN}${FINAL_TOKEN}`,
    );
    expect(resumeRequests).toBeGreaterThan(0);
    expect(conversationId).not.toBeNull();

    await page.unroute('**/api/chat/*/stream');
    await page.route('**/api/chat/*/stream', async (route) => {
      await route.fulfill({ status: 204 });
    });
    allowServerHistory = true;
    await page.reload();
    await expect(page.getByTestId('answer-text').last()).toContainText(
      `${FIRST_TOKEN}${FINAL_TOKEN}`,
    );
    await page.getByTestId('composer-model').click();
    await expect(
      page.getByTestId('model-free-badge-claude-sonnet-5'),
    ).toContainText('5/5');
    await page.screenshot({
      animations: 'disabled',
      path: `${EVIDENCE_DIR}/sponsored-conversation-preserved.png`,
    });
    await page.keyboard.press('Escape');
    expect(historyRequests).toBeGreaterThan(0);

    await firstLegServer.close();
  });

  test('explicit stop prevents a later refresh from resuming the stream', async ({
    page,
  }) => {
    // Given: an active stream has rendered a partial answer.
    await installHealthRoute(page);
    await installModelRoute(page);
    const activeServer = await startChatStreamServer({
      gapMs: LONG_GAP_MS,
      head: chunksForAnswer(STOP_TOKEN).slice(0, 3),
      tail: [],
    });
    const stopRequests: unknown[] = [];
    const resumeStatuses: number[] = [];
    const conversations: ConversationRow[] = [];
    let conversationId: null | string = null;

    await page.route('**/api/chat', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          body: JSON.stringify(conversations),
          contentType: 'application/json',
          status: 200,
        });
        return;
      }
      await route.fulfill({
        headers: { location: activeServer.url },
        status: 307,
      });
    });
    await page.route('**/api/chat/*', async (route) => {
      if (route.request().method() === 'PATCH') {
        conversationId = conversationIdFrom(route.request().url());
        if (
          conversations.every(
            (conversation) => conversation.id !== conversationId,
          )
        ) {
          conversations.unshift({
            id: conversationId,
            model: INFERENCE_MODEL,
            title: 'Започни долг одговор.',
          });
        }
        await route.fulfill({ status: 204 });
        return;
      }
      await route.fallback();
    });
    await page.route('**/api/chat/*/stop', async (route) => {
      stopRequests.push(JSON.parse(route.request().postData() ?? '{}'));
      await route.fulfill({
        body: JSON.stringify({ aborted: true, stopped: true }),
        contentType: 'application/json',
        status: 200,
      });
    });
    await page.route('**/api/chat/*/stream', async (route) => {
      resumeStatuses.push(204);
      await route.fulfill({ status: 204 });
    });
    await page.route('**/api/chat/*/history', async (route) => {
      await route.fulfill({
        body: emptyHistoryBody(conversationId),
        contentType: 'application/json',
        status: 200,
      });
    });

    await page.goto('/');
    await page.getByTestId('composer-input').fill('Започни долг одговор.');
    await page.getByTestId('composer-input').press('Enter');
    await expect(page.getByTestId('answer-text')).toContainText(STOP_TOKEN);

    // When: the user explicitly stops and refreshes the same chat.
    await page.getByTestId('composer-submit').click();
    await expect.poll(() => stopRequests).toHaveLength(1);
    const noActiveResponse = page.waitForResponse((response) =>
      CHAT_STREAM_URL_PATTERN.test(response.url()),
    );
    await page.reload();

    // Then: reconnect receives 204/no-active instead of replaying the stopped stream.
    const noActive = await noActiveResponse;
    expect(noActive.status()).toBe(204);
    expect(resumeStatuses.length).toBeGreaterThan(0);
    expect(resumeStatuses.every((status) => status === 204)).toBe(true);
    expect(stopRequests[0]).toEqual({
      activeStreamId: RESPONSE_ID,
    });

    await activeServer.close();
  });
});

/* eslint-enable camelcase -- end catalog wire fixtures. */

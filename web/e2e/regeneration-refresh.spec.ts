import { expect, test } from '@playwright/test';

import type { MyUIMessage } from '@/lib/api-types';
import type { ChatConversationHistory } from '@/lib/conversation-types';

import { installMockChatState } from './helpers/chat-state';
import { mockModels } from './helpers/models';
import { startChatStreamServer, type UiChunk } from './helpers/sse';

const CONVERSATION_ID = '11111111-2222-4333-8444-555555555555';
const RESPONSE_ID = '22222222-3333-4444-8555-666666666666';
const MODEL = 'claude-sonnet-5';
const TARGET_MESSAGE_ID = 'assistant-first';
const OLD_ANSWER = 'Стар прв одговор';
const NEW_ANSWER = 'Регенериран прв одговор';
const TRAILING_ANSWER = 'Стар втор одговор';

const userMessage = (id: string, text: string): MyUIMessage => ({
  id,
  metadata: {},
  parts: [{ text, type: 'text' }],
  role: 'user',
});

const assistantMessage = (
  id: string,
  responseId: string,
  text: string,
): MyUIMessage => ({
  id,
  metadata: { inferenceModel: MODEL, responseId },
  parts: [{ text, type: 'text' }],
  role: 'assistant',
});

const regenerationChunks = (answer: string): UiChunk[] => [
  {
    messageId: TARGET_MESSAGE_ID,
    messageMetadata: {
      inferenceModel: MODEL,
      replacementMessageId: TARGET_MESSAGE_ID,
      responseId: RESPONSE_ID,
    },
    type: 'start',
  },
  { id: 'txt-answer', type: 'text-start' },
  { delta: answer, id: 'txt-answer', type: 'text-delta' },
  { id: 'txt-answer', type: 'text-end' },
  { type: 'finish' },
];

test('refresh during regeneration replaces the target without restoring later turns', async ({
  page,
}) => {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      body: '{}',
      contentType: 'application/json',
      status: 200,
    });
  });
  await mockModels(page);
  const history: ChatConversationHistory = {
    conversation: {
      activeStream: null,
      id: CONVERSATION_ID,
      model: MODEL,
      title: 'Повеќе пораки',
    },
    messages: [
      userMessage('user-first', 'Прво прашање'),
      assistantMessage(TARGET_MESSAGE_ID, 'response-old-first', OLD_ANSWER),
      userMessage('user-second', 'Второ прашање'),
      assistantMessage(
        'assistant-second',
        'response-old-second',
        TRAILING_ANSWER,
      ),
    ],
  };
  const firstLegServer = await startChatStreamServer({
    gapMs: 30_000,
    head: regenerationChunks(NEW_ANSWER).slice(0, 3),
    tail: [],
  });

  try {
    let regenerationStarted = false;
    await installMockChatState(page, {
      conversations: [
        { id: CONVERSATION_ID, model: MODEL, title: 'Повеќе пораки' },
      ],
      histories: { [CONVERSATION_ID]: history },
      onCreate: () => {
        regenerationStarted = true;
      },
      streamUrl: firstLegServer.url,
    });
    await page.route('**/api/chat/*/history', async (route) => {
      await route.fulfill({
        body: JSON.stringify({
          ...history,
          conversation: {
            ...history.conversation,
            activeStream: regenerationStarted
              ? {
                  id: RESPONSE_ID,
                  replacementMessageId: TARGET_MESSAGE_ID,
                }
              : null,
          },
        }),
        contentType: 'application/json',
        status: 200,
      });
    });
    let resumeRequests = 0;
    await page.route('**/api/chat/*/stream', async (route) => {
      resumeRequests += 1;
      if (!regenerationStarted) {
        await route.fulfill({ status: 204 });
        return;
      }
      const body = regenerationChunks(NEW_ANSWER)
        .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
        .join('');
      await route.fulfill({
        body: `${body}data: [DONE]\n\n`,
        contentType: 'text/event-stream',
        headers: { 'x-vercel-ai-ui-message-stream': 'v1' },
        status: 200,
      });
    });

    await page.goto('/');
    await page
      .getByRole('button', { exact: true, name: 'Повеќе пораки' })
      .click();
    await expect(page.getByText(OLD_ANSWER)).toBeVisible();
    const regenerationRequest = page.waitForRequest(
      (request) =>
        request.url().endsWith('/api/chat') && request.method() === 'POST',
    );
    await page.getByRole('button', { name: 'Регенерирај' }).first().click();
    await regenerationRequest;

    await page.reload();

    await expect(page.getByText(NEW_ANSWER)).toBeVisible();
    await expect(page.getByText(OLD_ANSWER)).toHaveCount(0);
    await expect(page.getByText(TRAILING_ANSWER)).toHaveCount(0);
    expect(resumeRequests).toBeGreaterThan(0);
  } finally {
    await firstLegServer.close();
  }
});

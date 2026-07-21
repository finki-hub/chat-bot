import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ChatStateActiveStreamStatus } from '@/lib/chat-state-client';

/* eslint-disable camelcase -- Test fixtures mirror the Python chat state API wire format. */
import {
  CONVERSATION_ID,
  installRouteMocks,
  MODEL,
  resetRouteMocks,
  RESPONSE_ID,
  routeMocks,
  USER_ID,
} from './api-chat-route-support';

const STORED_TITLE = 'Stored title';

type ExpectedActiveStream = {
  readonly id: string;
  readonly replacementMessageId: null | string;
};

type StoredConversationOverrides = {
  readonly active_replacement_message_id?: null | string;
  readonly active_response_id?: null | string;
  readonly active_status?: ChatStateActiveStreamStatus | null;
  readonly active_stream_id?: null | string;
};

const importGet = async (): Promise<
  (
    req: Request,
    ctx: { readonly params: Promise<{ readonly id: string }> },
  ) => Promise<Response>
> => {
  const { GET } = await import('@/app/api/chat/[id]/history/route');

  return GET;
};

const historyRequest = (): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}/history`, {
    method: 'GET',
  });

const routeContext = () => ({
  params: Promise.resolve({ id: CONVERSATION_ID }),
});

const getHistory = async (): Promise<Response> =>
  (await importGet())(historyRequest(), routeContext());

const expectedConversation = (
  activeStream: ExpectedActiveStream | null = null,
) => ({
  activeStream,
  id: CONVERSATION_ID,
  model: MODEL,
  title: STORED_TITLE,
});

const storedConversation = (overrides: StoredConversationOverrides = {}) => ({
  active_replacement_message_id: null,
  active_response_id: null,
  active_status: null,
  active_stream_id: null,
  id: CONVERSATION_ID,
  model: MODEL,
  title: STORED_TITLE,
  user_id: USER_ID,
  ...overrides,
});

beforeEach(() => {
  vi.resetModules();
  resetRouteMocks();
  installRouteMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('GET /api/chat/[id]/history persisted messages', () => {
  it('returns persisted UI messages for the owner', async () => {
    // Given: the Python API has durable conversation messages for this owner.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: storedConversation(),
      messages: [
        {
          content: 'Stored question',
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d31',
          metadata: {},
          parts: null,
          response_id: null,
          role: 'user',
        },
        {
          content: 'Stored answer',
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d32',
          metadata: {
            diagnostics: {
              serverTotalMs: 1_700,
              serverTtftMs: 200,
              thinkingMs: 400,
            },
            inferenceModel: MODEL,
          },
          parts: [
            {
              state: 'done',
              text: 'Stored reasoning',
              type: 'reasoning',
            },
            { state: 'done', text: 'Stored answer', type: 'text' },
          ],
          response_id: RESPONSE_ID,
          role: 'assistant',
        },
      ],
    });
    // When: the browser asks for completed history after local storage was cleared.
    const res = await getHistory();

    // Then: the route exposes UI-message history with response id metadata preserved.
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual({
      conversation: expectedConversation(),
      messages: [
        {
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d31',
          metadata: {},
          parts: [{ text: 'Stored question', type: 'text' }],
          role: 'user',
        },
        {
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d32',
          metadata: {
            diagnostics: {
              serverTotalMs: 1_700,
              serverTtftMs: 200,
              thinkingMs: 400,
            },
            inferenceModel: MODEL,
            responseId: RESPONSE_ID,
            timing: { totalMs: 1_700, ttftMs: 200 },
          },
          parts: [
            {
              state: 'done',
              text: 'Stored reasoning',
              type: 'reasoning',
            },
            { state: 'done', text: 'Stored answer', type: 'text' },
          ],
          role: 'assistant',
        },
      ],
    });
    expect(routeMocks.stateClient.loadConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });
  });

  it('falls back to text when persisted parts are absent or invalid', async () => {
    // Given: legacy and malformed rows still have durable text content.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: storedConversation(),
      messages: [
        {
          content: 'Legacy question',
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d33',
          metadata: {},
          response_id: null,
          role: 'user',
        },
        {
          content: 'Recoverable answer',
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d34',
          metadata: {},
          parts: { text: 'Unexpected non-array parts' },
          response_id: RESPONSE_ID,
          role: 'assistant',
        },
      ],
    });

    // When: history is reconstructed from those rows.
    const res = await getHistory();

    // Then: neither turn is dropped and both use their durable text content.
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual({
      conversation: expectedConversation(),
      messages: [
        {
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d33',
          metadata: {},
          parts: [{ text: 'Legacy question', type: 'text' }],
          role: 'user',
        },
        {
          id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d34',
          metadata: { responseId: RESPONSE_ID },
          parts: [{ text: 'Recoverable answer', type: 'text' }],
          role: 'assistant',
        },
      ],
    });
  });
});

describe('GET /api/chat/[id]/history active streams', () => {
  it('returns the active regeneration descriptor needed after a refresh', async () => {
    const replacementMessageId = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d35';
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: storedConversation({
        active_replacement_message_id: replacementMessageId,
        active_response_id: RESPONSE_ID,
        active_status: 'streaming',
        active_stream_id: RESPONSE_ID,
      }),
      messages: [],
    });

    const res = await getHistory();

    await expect(res.json()).resolves.toStrictEqual({
      conversation: expectedConversation({
        id: RESPONSE_ID,
        replacementMessageId,
      }),
      messages: [],
    });
  });

  it.each(['completed', 'stopped', 'error'] as const)(
    'does not expose a stale active stream when its status is %s',
    async (activeStatus) => {
      // Given: terminal persistence still contains a stream id while cleanup races.
      routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
        conversation: storedConversation({
          active_replacement_message_id: RESPONSE_ID,
          active_response_id: RESPONSE_ID,
          active_status: activeStatus,
          active_stream_id: RESPONSE_ID,
        }),
        messages: [],
      });

      // When: the browser restores history after the terminal stream.
      const res = await getHistory();

      // Then: the HTTP response keeps the conversation shape but exposes no resume descriptor.
      expect(res.status).toBe(200);
      await expect(res.json()).resolves.toStrictEqual({
        conversation: expectedConversation(),
        messages: [],
      });
    },
  );

  it('does not expose history when the API rejects the user', async () => {
    // Given: the API state endpoint rejects this user for the conversation.
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');
    routeMocks.stateClient.loadConversation.mockRejectedValueOnce(
      new ChatStateRequestError(404),
    );

    // When: the authenticated user asks for inaccessible completed history.
    const res = await getHistory();

    // Then: the route preserves the ownership failure and returns no body.
    expect(res.status).toBe(404);
    await expect(res.text()).resolves.toBe('');
  });

  it('returns 401 when there is no authenticated session', async () => {
    const { AuthenticationRequiredError } =
      await import('@/lib/authenticated-chat-user');
    routeMocks.getAuthenticatedChatUserId.mockRejectedValueOnce(
      new AuthenticationRequiredError(),
    );

    const res = await getHistory();

    expect(res.status).toBe(401);
    expect(routeMocks.stateClient.loadConversation).not.toHaveBeenCalled();
  });
});

/* eslint-enable camelcase -- end Python chat state API wire fixtures. */

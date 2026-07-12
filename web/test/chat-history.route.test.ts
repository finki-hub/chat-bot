import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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

describe('GET /api/chat/[id]/history', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns persisted UI messages for the owner', async () => {
    // Given: the Python API has durable conversation messages for this owner.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: {
        active_response_id: null,
        active_status: null,
        active_stream_id: null,
        id: CONVERSATION_ID,
        model: MODEL,
        title: 'Stored title',
        user_id: USER_ID,
      },
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
    const get = await importGet();

    // When: the browser asks for completed history after local storage was cleared.
    const res = await get(historyRequest(), routeContext());
    const body: unknown = await res.json();

    // Then: the route exposes UI-message history with response id metadata preserved.
    expect(res.status).toBe(200);
    expect(body).toStrictEqual({
      conversation: {
        id: CONVERSATION_ID,
        model: MODEL,
        title: 'Stored title',
      },
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
      conversation: {
        active_response_id: null,
        active_status: null,
        active_stream_id: null,
        id: CONVERSATION_ID,
        model: MODEL,
        title: 'Stored title',
        user_id: USER_ID,
      },
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
          parts: [{ text: 42, type: 'text' }],
          response_id: RESPONSE_ID,
          role: 'assistant',
        },
      ],
    });

    // When: history is reconstructed from those rows.
    const res = await (await importGet())(historyRequest(), routeContext());
    const body: unknown = await res.json();

    // Then: neither turn is dropped and both use their durable text content.
    expect(res.status).toBe(200);
    expect(body).toStrictEqual({
      conversation: {
        id: CONVERSATION_ID,
        model: MODEL,
        title: 'Stored title',
      },
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

  it('does not expose history when the API rejects the user', async () => {
    // Given: the API state endpoint rejects this user for the conversation.
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');
    routeMocks.stateClient.loadConversation.mockRejectedValueOnce(
      new ChatStateRequestError(404),
    );

    // When: the authenticated user asks for inaccessible completed history.
    const res = await (await importGet())(historyRequest(), routeContext());

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

    const res = await (await importGet())(historyRequest(), routeContext());

    expect(res.status).toBe(401);
    expect(routeMocks.stateClient.loadConversation).not.toHaveBeenCalled();
  });
});

/* eslint-enable camelcase -- end Python chat state API wire fixtures. */

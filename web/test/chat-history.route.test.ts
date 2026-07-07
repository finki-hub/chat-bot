import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CONVERSATION_ID,
  installRouteMocks,
  OTHER_USER_ID,
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

const historyRequest = (userId = USER_ID): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}/history`, {
    headers: { 'X-Client-User-Id': userId },
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
    const get = await importGet();

    // When: the browser asks for completed history after local storage was cleared.
    const res = await get(historyRequest(), routeContext());
    const body: unknown = await res.json();

    // Then: the route exposes UI-message history with response id metadata preserved.
    expect(res.status).toBe(200);
    expect(body).toStrictEqual({
      conversation: {
        id: CONVERSATION_ID,
        model: 'claude-sonnet-4-6',
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
            inferenceModel: 'claude-sonnet-4-6',
            responseId: RESPONSE_ID,
          },
          parts: [{ text: 'Stored answer', type: 'text' }],
          role: 'assistant',
        },
      ],
    });
    expect(routeMocks.stateClient.loadConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });
  });

  it('does not expose history when the API rejects the user', async () => {
    // Given: the API state endpoint rejects this user for the conversation.
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');
    routeMocks.stateClient.loadConversation.mockRejectedValueOnce(
      new ChatStateRequestError(404),
    );

    // When: a different anonymous user asks for completed history.
    const res = await (
      await importGet()
    )(historyRequest(OTHER_USER_ID), routeContext());

    // Then: the route preserves the ownership failure and returns no body.
    expect(res.status).toBe(404);
    await expect(res.text()).resolves.toBe('');
  });
});

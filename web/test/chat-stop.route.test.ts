import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CONVERSATION_ID,
  installRouteMocks,
  MODEL,
  resetRouteMocks,
  RESPONSE_ID,
  routeMocks,
  USER_ID,
} from './api-chat-route-support';

/* eslint-disable camelcase -- Route tests mirror Python chat state API payloads. */

const NEWER_STREAM_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d23';

const importPost = async (): Promise<
  (
    req: Request,
    ctx: { readonly params: Promise<{ readonly id: string }> },
  ) => Promise<Response>
> => {
  const { POST } = await import('@/app/api/chat/[id]/stop/route');

  return POST;
};

const stopRequest = (body: object = {}): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}/stop`, {
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
    method: 'POST',
  });

const routeContext = () => ({
  params: Promise.resolve({ id: CONVERSATION_ID }),
});

describe('POST /api/chat/[id]/stop', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
    routeMocks.activeChatProducers.abort.mockReturnValue(true);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('is idempotent when the conversation has no active stream', async () => {
    // Given: the API state already has no active stream.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: {
        active_response_id: null,
        active_status: 'stopped',
        active_stream_id: null,
        id: CONVERSATION_ID,
        user_id: USER_ID,
      },
      messages: [],
    });

    // When: the browser repeats a stop request.
    const res = await (await importPost())(stopRequest(), routeContext());

    // Then: the route succeeds without aborting or clearing unrelated state.
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual({
      aborted: false,
      stopped: false,
    });
    expect(routeMocks.activeChatProducers.abort).not.toHaveBeenCalled();
    expect(
      routeMocks.stateClient.clearActiveStreamIfCurrent,
    ).not.toHaveBeenCalled();
  });

  it('does not clear a newer active stream when a stale stream id is supplied', async () => {
    // Given: a stale stop request races after a newer active stream was stored.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: {
        active_response_id: NEWER_STREAM_ID,
        active_status: 'streaming',
        active_stream_id: NEWER_STREAM_ID,
        id: CONVERSATION_ID,
        user_id: USER_ID,
      },
      messages: [],
    });

    // When: the stale stream id is stopped.
    const res = await (
      await importPost()
    )(stopRequest({ activeStreamId: RESPONSE_ID }), routeContext());

    // Then: the route reports stale success but leaves the newer stream untouched.
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual({
      aborted: false,
      stopped: false,
    });
    expect(routeMocks.activeChatProducers.abort).not.toHaveBeenCalled();
    expect(
      routeMocks.stateClient.stopActiveStreamIfCurrent,
    ).not.toHaveBeenCalled();
    expect(
      routeMocks.stateClient.clearActiveStreamIfCurrent,
    ).not.toHaveBeenCalled();
  });

  it('does not stop an active stream when the request omits the current stream id', async () => {
    // Given: a second tab has no response id, but this conversation has a live stream.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: {
        active_response_id: RESPONSE_ID,
        active_status: 'streaming',
        active_stream_id: RESPONSE_ID,
        id: CONVERSATION_ID,
        user_id: USER_ID,
      },
      messages: [],
    });

    // When: the stop body has no activeStreamId guard.
    const res = await (
      await importPost()
    )(
      stopRequest({ assistantSnapshot: { content: 'stale', metadata: {} } }),
      routeContext(),
    );

    // Then: the live stream remains untouched.
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual({
      aborted: false,
      stopped: false,
    });
    expect(routeMocks.activeChatProducers.abort).not.toHaveBeenCalled();
    expect(
      routeMocks.stateClient.upsertAssistantMessage,
    ).not.toHaveBeenCalled();
    expect(
      routeMocks.stateClient.clearActiveStreamIfCurrent,
    ).not.toHaveBeenCalled();
  });

  it('persists a supplied assistant snapshot before stopping the current stream', async () => {
    // Given: an active stream has visible partial assistant content in the browser.
    const snapshot = {
      content: 'Partial answer',
      metadata: { inferenceModel: MODEL },
    };

    // When: the browser explicitly stops the current stream with that snapshot.
    const res = await (
      await importPost()
    )(
      stopRequest({ activeStreamId: RESPONSE_ID, assistantSnapshot: snapshot }),
      routeContext(),
    );

    // Then: partial content is persisted, the producer is aborted, and active state is current-guard cleared.
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual({
      aborted: true,
      stopped: true,
    });
    expect(routeMocks.stateClient.upsertAssistantMessage).toHaveBeenCalledWith({
      content: 'Partial answer',
      conversationId: CONVERSATION_ID,
      metadata: {
        inferenceModel: MODEL,
        responseId: RESPONSE_ID,
        stopped: true,
      },
      responseId: RESPONSE_ID,
      userId: USER_ID,
    });
    expect(routeMocks.activeChatProducers.abort).toHaveBeenCalledWith(
      RESPONSE_ID,
    );
    expect(
      routeMocks.stateClient.stopActiveStreamIfCurrent,
    ).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      streamId: RESPONSE_ID,
      userId: USER_ID,
    });
    expect(
      routeMocks.stateClient.clearActiveStreamIfCurrent,
    ).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      streamId: RESPONSE_ID,
      userId: USER_ID,
    });
  });

  it('returns 401 when there is no authenticated session', async () => {
    const { AuthenticationRequiredError } =
      await import('@/lib/authenticated-chat-user');
    routeMocks.getAuthenticatedChatUserId.mockRejectedValueOnce(
      new AuthenticationRequiredError(),
    );

    const res = await (await importPost())(stopRequest(), routeContext());

    expect(res.status).toBe(401);
    expect(routeMocks.stateClient.loadConversation).not.toHaveBeenCalled();
  });
});

/* eslint-enable camelcase -- end Python chat state API payload fixtures. */

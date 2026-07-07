import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CONVERSATION_ID,
  installRouteMocks,
  OTHER_USER_ID,
  resetRouteMocks,
  RESPONSE_ID,
  routeMocks,
  sseBody,
  USER_ID,
} from './api-chat-route-support';

/* eslint-disable camelcase -- Route tests mirror Python chat state API payloads. */

const importGet = async (): Promise<
  (
    req: Request,
    ctx: { readonly params: Promise<{ readonly id: string }> },
  ) => Promise<Response>
> => {
  const { GET } = await import('@/app/api/chat/[id]/stream/route');

  return GET;
};

const streamRequest = (userId = USER_ID): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}/stream`, {
    headers: { 'X-Client-User-Id': userId },
    method: 'GET',
  });

const routeContext = () => ({
  params: Promise.resolve({ id: CONVERSATION_ID }),
});

describe('GET /api/chat/[id]/stream', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns UI-message SSE when the owner has an active Redis stream', async () => {
    // Given: the API state points at a live Redis-backed UI message stream.
    routeMocks.resumableContext.resumeExistingStream.mockResolvedValueOnce(
      sseBody('data: owner-token\n\n'),
    );

    // When: the owner reconnects to the conversation stream endpoint.
    const res = await (await importGet())(streamRequest(), routeContext());

    // Then: the route returns the resumable UI-message SSE surface.
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/event-stream');
    await expect(res.text()).resolves.toContain('owner-token');
    expect(routeMocks.stateClient.loadConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });
    expect(
      routeMocks.resumableContext.resumeExistingStream,
    ).toHaveBeenCalledWith(RESPONSE_ID);
  });

  it('allows two browser surfaces to reconnect to the same active stream', async () => {
    // Given: two tabs for the same owner both ask for the current active stream.
    routeMocks.resumableContext.resumeExistingStream
      .mockResolvedValueOnce(sseBody('data: tab-one\n\n'))
      .mockResolvedValueOnce(sseBody('data: tab-two\n\n'));

    // When: both tab-level reconnect requests hit the route.
    const get = await importGet();
    const [first, second] = await Promise.all([
      get(streamRequest(), routeContext()),
      get(streamRequest(), routeContext()),
    ]);

    // Then: neither reconnect steals or clears active state from the other.
    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    await expect(first.text()).resolves.toContain('tab-one');
    await expect(second.text()).resolves.toContain('tab-two');
    expect(
      routeMocks.resumableContext.resumeExistingStream,
    ).toHaveBeenCalledTimes(2);
    expect(
      routeMocks.stateClient.clearActiveStreamIfCurrent,
    ).not.toHaveBeenCalled();
  });

  it('returns 204 when the owner has no active stream', async () => {
    // Given: the API state has no active stream for this owned conversation.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
      conversation: {
        active_response_id: null,
        active_status: null,
        active_stream_id: null,
        id: CONVERSATION_ID,
        user_id: USER_ID,
      },
      messages: [],
    });

    // When: the owner reconnects.
    const res = await (await importGet())(streamRequest(), routeContext());

    // Then: the route reports no stream without touching Redis.
    expect(res.status).toBe(204);
    await expect(res.text()).resolves.toBe('');
    expect(
      routeMocks.resumableContext.resumeExistingStream,
    ).not.toHaveBeenCalled();
  });

  it('does not expose a stream body when the API rejects the user', async () => {
    // Given: the API state endpoint rejects this user for the conversation.
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');
    routeMocks.stateClient.loadConversation.mockRejectedValueOnce(
      new ChatStateRequestError(404),
    );

    // When: a different anonymous user attempts to resume.
    const res = await (
      await importGet()
    )(streamRequest(OTHER_USER_ID), routeContext());

    // Then: the route preserves the ownership failure and returns no SSE body.
    expect(res.status).toBe(404);
    await expect(res.text()).resolves.toBe('');
    expect(
      routeMocks.resumableContext.resumeExistingStream,
    ).not.toHaveBeenCalled();
  });

  it('clears stale API active state and returns 204 when Redis expired', async () => {
    // Given: API state points at a stream id that no longer exists in Redis.
    routeMocks.resumableContext.resumeExistingStream.mockResolvedValueOnce(
      null,
    );

    // When: the owner reconnects after Redis expiry.
    const res = await (await importGet())(streamRequest(), routeContext());

    // Then: stale active state is cleared only for that current stream id.
    expect(res.status).toBe(204);
    expect(
      routeMocks.stateClient.clearActiveStreamIfCurrent,
    ).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      streamId: RESPONSE_ID,
      userId: USER_ID,
    });
  });
});

/* eslint-enable camelcase -- end Python chat state API payload fixtures. */

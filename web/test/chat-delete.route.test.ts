import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CONVERSATION_ID,
  installRouteMocks,
  resetRouteMocks,
  routeMocks,
  USER_ID,
} from './api-chat-route-support';

const importDelete = async (): Promise<
  (
    req: Request,
    ctx: { readonly params: Promise<{ readonly id: string }> },
  ) => Promise<Response>
> => {
  const { DELETE } = await import('@/app/api/chat/[id]/route');

  return DELETE;
};

const importPatch = async (): Promise<
  (
    req: Request,
    ctx: { readonly params: Promise<{ readonly id: string }> },
  ) => Promise<Response>
> => {
  const { PATCH } = await import('@/app/api/chat/[id]/route');

  return PATCH;
};

const deleteRequest = (): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}`, {
    method: 'DELETE',
  });

const patchRequest = (body: unknown): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}`, {
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
    method: 'PATCH',
  });

const routeContext = () => ({
  params: Promise.resolve({ id: CONVERSATION_ID }),
});

describe('DELETE /api/chat/[id]', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('deletes persisted conversation state for the authenticated owner', async () => {
    const res = await (await importDelete())(deleteRequest(), routeContext());

    expect(res.status).toBe(204);
    expect(routeMocks.stateClient.deleteConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });
  });

  it('returns 401 when there is no authenticated session', async () => {
    const { AuthenticationRequiredError } =
      await import('@/lib/authenticated-chat-user');
    routeMocks.getAuthenticatedChatUserId.mockRejectedValueOnce(
      new AuthenticationRequiredError(),
    );

    const res = await (await importDelete())(deleteRequest(), routeContext());

    expect(res.status).toBe(401);
    expect(routeMocks.stateClient.deleteConversation).not.toHaveBeenCalled();
  });

  it('preserves upstream ownership failures', async () => {
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');
    routeMocks.stateClient.deleteConversation.mockRejectedValueOnce(
      new ChatStateRequestError(404),
    );

    const res = await (await importDelete())(deleteRequest(), routeContext());

    expect(res.status).toBe(404);
  });

  it('updates conversation title for the authenticated owner', async () => {
    const res = await (
      await importPatch()
    )(patchRequest({ title: 'Renamed' }), routeContext());

    expect(res.status).toBe(204);
    expect(routeMocks.stateClient.updateConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      title: 'Renamed',
      userId: USER_ID,
    });
  });

  it('creates a conversation when the browser starts a new server chat', async () => {
    const res = await (
      await importPatch()
    )(
      patchRequest({ model: 'claude-sonnet-5', title: 'New chat' }),
      routeContext(),
    );

    expect(res.status).toBe(204);
    expect(routeMocks.stateClient.upsertConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      model: 'claude-sonnet-5',
      title: 'New chat',
      userId: USER_ID,
    });
    expect(routeMocks.stateClient.updateConversation).not.toHaveBeenCalled();
  });

  it('skips generated title updates when the current title changed', async () => {
    const res = await (
      await importPatch()
    )(
      patchRequest({ expectedTitle: 'Different', title: 'Generated' }),
      routeContext(),
    );

    expect(res.status).toBe(204);
    expect(routeMocks.stateClient.loadConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });
    expect(routeMocks.stateClient.updateConversation).not.toHaveBeenCalled();
  });
});

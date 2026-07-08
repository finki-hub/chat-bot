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

const deleteRequest = (): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}`, {
    method: 'DELETE',
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
});

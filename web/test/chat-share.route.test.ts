import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CONVERSATION_ID,
  installRouteMocks,
  resetRouteMocks,
  routeMocks,
  SHARE_TOKEN,
  USER_ID,
} from './api-chat-route-support';

type ShareRoutePost = (
  request: Request,
  context: { readonly params: Promise<{ readonly id: string }> },
) => Promise<Response>;

const importPost = async (): Promise<ShareRoutePost> => {
  const route = await import('@/app/api/chat/[id]/share/route');
  return route.POST;
};

const request = (): Request =>
  new Request(`http://localhost/api/chat/${CONVERSATION_ID}/share`, {
    method: 'POST',
  });

const context = () => ({ params: Promise.resolve({ id: CONVERSATION_ID }) });

describe('POST /api/chat/[id]/share', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
  });

  it('creates a stable share token for the authenticated owner', async () => {
    const response = await (await importPost())(request(), context());

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toStrictEqual({
      shareToken: SHARE_TOKEN,
    });
    expect(
      routeMocks.sharingClient.createConversationShare,
    ).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      userId: USER_ID,
    });
  });

  it('returns 401 without an authenticated owner', async () => {
    const { AuthenticationRequiredError } =
      await import('@/lib/authenticated-chat-user');
    routeMocks.getAuthenticatedChatUserId.mockRejectedValueOnce(
      new AuthenticationRequiredError(),
    );

    const response = await (await importPost())(request(), context());

    expect(response.status).toBe(401);
    expect(
      routeMocks.sharingClient.createConversationShare,
    ).not.toHaveBeenCalled();
  });

  it('preserves ownership failures from the chat-state API', async () => {
    const { ChatStateRequestError } = await import('@/lib/chat-state-client');
    routeMocks.sharingClient.createConversationShare.mockRejectedValueOnce(
      new ChatStateRequestError(404),
    );

    const response = await (await importPost())(request(), context());

    expect(response.status).toBe(404);
    await expect(response.text()).resolves.toBe('');
  });
});

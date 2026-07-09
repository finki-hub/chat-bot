import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CONVERSATION_ID,
  installRouteMocks,
  resetRouteMocks,
  routeMocks,
  USER_ID,
} from './api-chat-route-support';

const importRoutes = async (): Promise<{
  readonly DELETE: () => Promise<Response>;
  readonly GET: () => Promise<Response>;
}> => {
  const { DELETE, GET } = await import('@/app/api/chat/route');

  return { DELETE, GET };
};

describe('GET /api/chat', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('lists authenticated server conversations for the sidebar', async () => {
    const { GET } = await importRoutes();

    const res = await GET();

    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toStrictEqual([
      {
        id: CONVERSATION_ID,
        model: 'claude-sonnet-4-6',
        title: 'Stored title',
      },
    ]);
    expect(routeMocks.stateClient.listConversations).toHaveBeenCalledWith({
      userId: USER_ID,
    });
  });

  it('clears authenticated server conversations', async () => {
    const { DELETE } = await importRoutes();

    const res = await DELETE();

    expect(res.status).toBe(204);
    expect(routeMocks.stateClient.clearConversations).toHaveBeenCalledWith({
      userId: USER_ID,
    });
  });
});

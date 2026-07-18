import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/* eslint-disable camelcase -- fixtures mirror the Python API wire contract. */
import {
  API_BASE_URL,
  CONVERSATION_ID,
  installRouteMocks,
  JSON_CONTENT_TYPE,
  MODEL,
  resetRouteMocks,
  RESPONSE_ID,
  routeMocks,
  sseBody,
  USER_ID,
} from './api-chat-route-support';

const importPost = async (): Promise<(req: Request) => Promise<Response>> => {
  const { POST } = await import('@/app/api/chat/route');
  return POST;
};

describe('POST /api/chat user metadata ownership', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response(
          sseBody(
            'event: token\ndata: {"text":"Одговор"}\n\n',
            'event: done\ndata: {}\n\n',
          ),
          {
            headers: {
              'content-type': 'text/event-stream',
              'X-Response-Id': RESPONSE_ID,
            },
            status: 200,
          },
        ),
      ),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('uses the authenticated owner for upstream and persisted assistant metadata', async () => {
    const post = await importPost();
    const response = await post(
      new Request('http://localhost/api/chat', {
        body: JSON.stringify({
          id: CONVERSATION_ID,
          messages: [
            {
              id: 'u1',
              parts: [{ text: 'Прашање', type: 'text' }],
              role: 'user',
            },
          ],
          model: MODEL,
          sponsored_quota: { limit: 5, remaining: 0 },
          user_id: 'forged-browser-user',
        }),
        headers: { 'content-type': JSON_CONTENT_TYPE },
        method: 'POST',
      }),
    );
    await response.text();

    const fetchMock = vi.mocked(fetch);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      readonly sponsored_quota?: unknown;
      readonly user_id?: string;
    };

    expect(url).toBe(`${API_BASE_URL}/chat/`);
    expect(body.user_id).toBe(USER_ID);
    expect(body.sponsored_quota).toBeUndefined();
    expect(routeMocks.stateClient.upsertConversation).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      model: MODEL,
      userId: USER_ID,
    });
    expect(routeMocks.stateClient.upsertUserMessage).toHaveBeenCalledWith({
      content: 'Прашање',
      conversationId: CONVERSATION_ID,
      messageId: 'u1',
      userId: USER_ID,
    });
    expect(routeMocks.stateClient.upsertAssistantMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: CONVERSATION_ID,
        metadata: { inferenceModel: MODEL, responseId: RESPONSE_ID },
        responseId: RESPONSE_ID,
        userId: USER_ID,
      }),
    );
  });
});

/* eslint-enable camelcase -- end wire-contract fixtures. */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/* eslint-disable camelcase -- fixtures mirror the Python API wire contract. */
import {
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

const stored = (id: string, role: 'assistant' | 'user', content: string) => ({
  content,
  id,
  metadata: {},
  response_id: role === 'assistant' ? id : null,
  role,
});

const userTurn = (id: string, text: string) => ({
  id,
  parts: [{ text, type: 'text' }],
  role: 'user',
});

const loadStored = (): void => {
  routeMocks.stateClient.loadConversation.mockResolvedValueOnce({
    conversation: {
      active_replacement_message_id: null,
      active_response_id: null,
      active_status: null,
      active_stream_id: null,
      id: CONVERSATION_ID,
      model: MODEL,
      title: 'Stored title',
      user_id: USER_ID,
    },
    messages: [
      stored('u1', 'user', 'First question'),
      stored('a1', 'assistant', 'First answer'),
      stored('u2', 'user', 'Second question'),
      stored('a2', 'assistant', 'Second answer'),
    ],
  });
};

const post = async (body: Record<string, unknown>) => {
  const { POST } = await import('@/app/api/chat/route');

  return POST(
    new Request('http://localhost/api/chat', {
      body: JSON.stringify({ id: CONVERSATION_ID, model: MODEL, ...body }),
      headers: { 'content-type': JSON_CONTENT_TYPE },
      method: 'POST',
    }),
  );
};

const upstreamMessages = (): unknown => {
  const init = vi.mocked(fetch).mock.calls[0]?.[1];

  return (JSON.parse(init?.body as string) as { messages: unknown }).messages;
};

describe('POST /api/chat editing a question', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response(sseBody('event: done\ndata: {}\n\n'), {
          headers: {
            'content-type': 'text/event-stream',
            'X-Response-Id': RESPONSE_ID,
          },
          status: 200,
        }),
      ),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('drops what followed the edited question before saving it', async () => {
    loadStored();

    const response = await post({
      messageId: 'u1',
      messages: [userTurn('u1', 'Edited first question')],
      trigger: 'submit-message',
    });
    await response.text();

    const prune = routeMocks.stateClient.pruneAfterUserMessage;
    const upsert = routeMocks.stateClient.upsertUserMessage;

    expect(prune).toHaveBeenCalledWith({
      conversationId: CONVERSATION_ID,
      messageId: 'u1',
      userId: USER_ID,
    });
    expect(upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        content: 'Edited first question',
        messageId: 'u1',
      }),
    );
    expect(prune.mock.invocationCallOrder[0]).toBeLessThan(
      upsert.mock.invocationCallOrder[0] ?? 0,
    );
    // The model sees no history from after the edited question.
    expect(upstreamMessages()).toStrictEqual([
      { content: 'Edited first question', role: 'user' },
    ]);
  });

  it('prunes nothing for a new question', async () => {
    loadStored();

    const response = await post({
      messages: [userTurn('u3', 'Third question')],
      trigger: 'submit-message',
    });
    await response.text();

    expect(routeMocks.stateClient.pruneAfterUserMessage).not.toHaveBeenCalled();
    expect(routeMocks.stateClient.upsertUserMessage).toHaveBeenCalledOnce();
  });
});

/* eslint-enable camelcase -- fixture scope ends here. */

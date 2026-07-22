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

const persistedMessage = (
  id: string,
  role: 'assistant' | 'user',
  content: string,
) => ({
  content,
  id,
  metadata: {},
  response_id: role === 'assistant' ? id : null,
  role,
});

describe('POST /api/chat regeneration target', () => {
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

  it('rejects a persisted user message used as the regeneration target', async () => {
    // Given a browser request that points regeneration at the second persisted user.
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
        persistedMessage('u1', 'user', 'First question'),
        persistedMessage('a1', 'assistant', 'First answer'),
        persistedMessage('u2', 'user', 'Second question'),
      ],
    });
    const request = new Request('http://localhost/api/chat', {
      body: JSON.stringify({
        id: CONVERSATION_ID,
        messageId: 'u2',
        messages: [
          {
            id: 'u1',
            parts: [{ text: 'First question', type: 'text' }],
            role: 'user',
          },
          {
            id: 'a1',
            parts: [{ text: 'First answer', type: 'text' }],
            role: 'assistant',
          },
          {
            id: 'u2',
            parts: [{ text: 'Second question', type: 'text' }],
            role: 'user',
          },
        ],
        model: MODEL,
        trigger: 'regenerate-message',
      }),
      headers: { 'content-type': JSON_CONTENT_TYPE },
      method: 'POST',
    });

    // When the BFF validates the target against server-owned state.
    const { POST } = await import('@/app/api/chat/route');
    const response = await POST(request);
    const body = await response.text();

    // Then it returns a stream error without starting persistence or inference.
    expect(body).toContain('data-error');
    expect(fetch).not.toHaveBeenCalled();
    expect(routeMocks.stateClient.upsertUserMessage).not.toHaveBeenCalled();
    expect(routeMocks.stateClient.setActiveStream).not.toHaveBeenCalled();
  });
});

/* eslint-enable camelcase -- fixture scope ends here. */

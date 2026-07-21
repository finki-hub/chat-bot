import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  API_BASE_URL,
  CONVERSATION_ID,
  installRouteMocks,
  JSON_CONTENT_TYPE,
  type LoadedConversation,
  MODEL,
  resetRouteMocks,
  RESPONSE_ID,
  routeMocks,
  sseBody,
  USER_ID,
} from './api-chat-route-support';

const okStreamResponse = (): Response =>
  new Response(sseBody('event: done\ndata: {}\n\n'), {
    headers: {
      'content-type': 'text/event-stream',
      'X-Response-Id': RESPONSE_ID,
    },
    status: 200,
  });

const ACTIVE_RESPONSE_ID = 'active_response_id' as const;
const ACTIVE_REPLACEMENT_MESSAGE_ID = 'active_replacement_message_id' as const;
const ACTIVE_STATUS = 'active_status' as const;
const ACTIVE_STREAM_ID = 'active_stream_id' as const;
const RESPONSE_ID_FIELD = 'response_id' as const;
const USER_ID_FIELD = 'user_id' as const;

const importPost = async (): Promise<(req: Request) => Promise<Response>> => {
  const { POST } = await import('@/app/api/chat/route');

  return POST;
};

const loadedConversation = (
  messages: LoadedConversation['messages'],
): LoadedConversation => ({
  conversation: {
    [ACTIVE_REPLACEMENT_MESSAGE_ID]: null,
    [ACTIVE_RESPONSE_ID]: RESPONSE_ID,
    [ACTIVE_STATUS]: 'streaming',
    [ACTIVE_STREAM_ID]: RESPONSE_ID,
    id: CONVERSATION_ID,
    model: MODEL,
    title: 'Stored title',
    [USER_ID_FIELD]: USER_ID,
  },
  messages,
});

const persistedMessage = (
  id: string,
  role: 'assistant' | 'user',
  content: string,
) => ({
  content,
  id,
  metadata: {},
  [RESPONSE_ID_FIELD]: role === 'assistant' ? id : null,
  role,
});

const chatRequest = (body: object): Request =>
  new Request('http://localhost/api/chat', {
    body: JSON.stringify(body),
    headers: { 'content-type': JSON_CONTENT_TYPE },
    method: 'POST',
  });

describe('POST /api/chat boundary regressions', () => {
  beforeEach(() => {
    vi.resetModules();
    resetRouteMocks();
    installRouteMocks();
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>().mockResolvedValue(okStreamResponse()),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('caps and truncates persisted history before forwarding to the Python API', async () => {
    // Given: persisted state exceeds the API's 10-turn / 2000-char contract.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce(
      loadedConversation(
        Array.from({ length: 12 }, (_, index) =>
          persistedMessage(
            `stored-${index}`,
            index % 2 === 0 ? 'user' : 'assistant',
            `stored-${index}-${'x'.repeat(2_100)}`,
          ),
        ),
      ),
    );

    // When: the browser sends the next user turn.
    const response = await (
      await importPost()
    )(
      chatRequest({
        id: CONVERSATION_ID,
        messages: [
          {
            id: 'current-user',
            parts: [{ text: 'Current question', type: 'text' }],
            role: 'user',
          },
        ],
        model: MODEL,
      }),
    );

    await response.text();

    // Then: the upstream payload remains valid for api/app/schemas/chat.py.
    const fetchMock = vi.mocked(fetch);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const sentBody = JSON.parse(init.body as string) as {
      readonly messages: ReadonlyArray<{
        readonly content: string;
        readonly role: string;
      }>;
    };

    expect(url).toBe(`${API_BASE_URL}/chat/`);
    expect(sentBody.messages).toHaveLength(10);
    expect(sentBody.messages.at(-1)).toStrictEqual({
      content: 'Current question',
      role: 'user',
    });
    expect(sentBody.messages.slice(0, -1)).toStrictEqual(
      Array.from({ length: 9 }, (_, index) => {
        const persistedIndex = index + 3;

        return {
          content: `stored-${persistedIndex}-${'x'.repeat(2_100)}`.slice(
            0,
            2_000,
          ),
          role: persistedIndex % 2 === 0 ? 'user' : 'assistant',
        };
      }),
    );
  });

  it('uses the regenerate request window for both persistence and upstream history', async () => {
    // Given: the browser is regenerating the first assistant in a longer conversation.
    routeMocks.stateClient.loadConversation.mockResolvedValueOnce(
      loadedConversation([
        persistedMessage('u1', 'user', 'Original question'),
        persistedMessage('a1', 'assistant', 'Old answer'),
        persistedMessage('u2', 'user', 'Future question'),
        persistedMessage('a2', 'assistant', 'Future answer'),
      ]),
    );

    // When: the chat route receives the full UI conversation plus the regenerate target.
    const response = await (
      await importPost()
    )(
      chatRequest({
        id: CONVERSATION_ID,
        messageId: 'a1',
        messages: [
          {
            id: 'u1',
            parts: [{ text: 'Original question', type: 'text' }],
            role: 'user',
          },
          {
            id: 'a1',
            parts: [{ text: 'Old answer', type: 'text' }],
            role: 'assistant',
          },
          {
            id: 'u2',
            parts: [{ text: 'Future question', type: 'text' }],
            role: 'user',
          },
          {
            id: 'a2',
            parts: [{ text: 'Future answer', type: 'text' }],
            role: 'assistant',
          },
        ],
        model: MODEL,
        trigger: 'regenerate-message',
      }),
    );

    await response.text();

    // Then: the route persists and forwards only the same current user turn chosen by toChatRequestBody.
    expect(routeMocks.stateClient.upsertUserMessage).toHaveBeenCalledWith({
      content: 'Original question',
      conversationId: CONVERSATION_ID,
      messageId: 'u1',
      userId: USER_ID,
    });

    const fetchMock = vi.mocked(fetch);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const sentBody = JSON.parse(init.body as string) as {
      readonly messages: ReadonlyArray<{
        readonly content: string;
        readonly role: string;
      }>;
    };

    expect(sentBody.messages).toStrictEqual([
      { content: 'Original question', role: 'user' },
    ]);
  });
});

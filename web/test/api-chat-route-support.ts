import { vi } from 'vitest';

/* eslint-disable camelcase -- Route tests mirror Python chat state API payloads. */
/* eslint-disable @typescript-eslint/consistent-type-imports -- Vitest importOriginal type annotation needs inline module type. */
/* eslint-disable @typescript-eslint/no-unused-vars -- Test doubles preserve production call signatures. */
/* eslint-disable @typescript-eslint/require-await -- Test doubles mimic async production APIs. */
/* eslint-disable sonarjs/void-use -- Explicitly marks intentionally ignored test-only parameter. */

export const API_BASE_URL = 'https://api:8880';
export const CONVERSATION_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d21';
export const JSON_CONTENT_TYPE = 'application/json';
export const MODEL = 'claude-sonnet-4-6';
export const RESPONSE_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d22';
export const USER_ID = 'anon-user-1';
export const OTHER_USER_ID = 'anon-user-2';

export type LoadedConversation = {
  readonly conversation: {
    readonly active_response_id: null | string;
    readonly active_status: null | string;
    readonly active_stream_id: null | string;
    readonly id: string;
    readonly model?: null | string;
    readonly title?: null | string;
    readonly user_id: string;
  };
  readonly messages: readonly unknown[];
};

type ResumeExistingStream = (
  streamId: string,
) => Promise<null | ReadableStream<Uint8Array> | undefined>;

type StateClientInput = {
  readonly activeResponseId?: string;
  readonly activeStreamId?: string;
  readonly content?: string;
  readonly conversationId: string;
  readonly messageId?: string;
  readonly metadata?: Record<string, unknown>;
  readonly model?: string;
  readonly responseId?: string;
  readonly retainedMessageIds?: readonly string[];
  readonly streamId?: string;
  readonly userId: string;
};

const readStringStream = async (
  stream: ReadableStream<string>,
): Promise<string> => {
  const reader = stream.getReader();
  let out = '';

  try {
    for (;;) {
      const next = await reader.read();

      if (next.done) {
        return out;
      }

      out += next.value;
    }
  } finally {
    reader.releaseLock();
  }
};

export const sseBody = (...frames: string[]): ReadableStream<Uint8Array> => {
  const enc = new TextEncoder();

  return new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(enc.encode(frame));
      }

      controller.close();
    },
  });
};

export const routeMocks = {
  activeChatProducers: {
    abort: vi.fn(),
    register: vi.fn(),
    unregister: vi.fn(),
  },
  consumedResumableStreams: [] as string[],
  createChatResumableStreamContext: vi.fn(),
  createChatStateClient: vi.fn(),
  getAuthenticatedChatUserId: vi.fn(async () => USER_ID),
  resumableContext: {
    createNewResumableStream: vi.fn(
      async (streamId: string, makeStream: () => ReadableStream<string>) => {
        const stream = makeStream();
        routeMocks.consumedResumableStreams.push(
          await readStringStream(stream),
        );
        void streamId;

        return stream;
      },
    ),
    resumeExistingStream: vi.fn<ResumeExistingStream>(async (_streamId) =>
      sseBody('data: resumed\n\n'),
    ),
  },
  stateClient: {
    clearActiveStreamIfCurrent: vi.fn(async (_input: StateClientInput) => {}),
    clearConversations: vi.fn(
      async (_input: { readonly userId: string }) => {},
    ),
    deleteConversation: vi.fn(async (_input: StateClientInput) => {}),
    listConversations: vi.fn(async (_input: { readonly userId: string }) => [
      {
        active_response_id: null,
        active_status: null,
        active_stream_id: null,
        id: CONVERSATION_ID,
        model: MODEL,
        title: 'Stored title',
        user_id: USER_ID,
      },
    ]),
    loadConversation: vi.fn(
      async (_input: StateClientInput): Promise<LoadedConversation> => ({
        conversation: {
          active_response_id: RESPONSE_ID,
          active_status: 'streaming',
          active_stream_id: RESPONSE_ID,
          id: CONVERSATION_ID,
          model: MODEL,
          title: 'Stored title',
          user_id: USER_ID,
        },
        messages: [
          {
            content: 'Stored question',
            id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d31',
            metadata: {},
            response_id: null,
            role: 'user',
          },
          {
            content: 'Stored answer',
            id: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d32',
            metadata: { inferenceModel: MODEL },
            response_id: RESPONSE_ID,
            role: 'assistant',
          },
        ],
      }),
    ),
    replaceAssistantMessage: vi.fn(async (_input: StateClientInput) => {}),
    setActiveStream: vi.fn(async (_input: StateClientInput) => {}),
    stopActiveStreamIfCurrent: vi.fn(async (_input: StateClientInput) => {}),
    updateConversation: vi.fn(async (_input: StateClientInput) => {}),
    upsertAssistantMessage: vi.fn(async (_input: StateClientInput) => {}),
    upsertConversation: vi.fn(async (_input: StateClientInput) => {}),
    upsertUserMessage: vi.fn(async (_input: StateClientInput) => {}),
  },
};

export const resetRouteMocks = (): void => {
  vi.clearAllMocks();
  routeMocks.consumedResumableStreams.length = 0;
  routeMocks.getAuthenticatedChatUserId.mockResolvedValue(USER_ID);
  routeMocks.createChatStateClient.mockReturnValue(routeMocks.stateClient);
  routeMocks.createChatResumableStreamContext.mockReturnValue(
    routeMocks.resumableContext,
  );
};

export const installRouteMocks = (): void => {
  vi.doMock('@/lib/env', () => ({
    API_BASE_URL,
    CHAT_API_KEY: 'test-key',
    env: { API_BASE_URL, CHAT_API_KEY: 'test-key' },
  }));
  vi.doMock('@/lib/authenticated-chat-user', () => {
    class AuthenticationRequiredError extends Error {
      constructor() {
        super('Authentication required');
        this.name = 'AuthenticationRequiredError';
      }
    }

    return {
      AuthenticationRequiredError,
      getAuthenticatedChatUserId: routeMocks.getAuthenticatedChatUserId,
    };
  });
  vi.doMock('@/lib/chat-state-client', async (importOriginal) => {
    const actual =
      await importOriginal<typeof import('@/lib/chat-state-client')>();

    return {
      ...actual,
      createChatStateClient: routeMocks.createChatStateClient,
    };
  });
  vi.doMock('@/lib/resumable-stream-context', async (importOriginal) => {
    const actual =
      await importOriginal<typeof import('@/lib/resumable-stream-context')>();

    return {
      ...actual,
      activeChatProducers: routeMocks.activeChatProducers,
      createChatResumableStreamContext:
        routeMocks.createChatResumableStreamContext,
    };
  });
};

/* eslint-enable sonarjs/void-use -- end intentional test-only ignored parameter. */
/* eslint-enable @typescript-eslint/require-await -- end async production API test doubles. */
/* eslint-enable @typescript-eslint/no-unused-vars -- end production call signature test doubles. */
/* eslint-enable @typescript-eslint/consistent-type-imports -- end Vitest importOriginal type annotations. */
/* eslint-enable camelcase -- end Python chat state API payload fixtures. */

export const chatRequest = (
  overrides: { readonly signal?: AbortSignal; readonly text?: string } = {},
): Request =>
  new Request('http://localhost/api/chat', {
    body: JSON.stringify({
      id: CONVERSATION_ID,
      messages: [
        {
          id: 'u1',
          parts: [{ text: overrides.text ?? 'Здраво', type: 'text' }],
          role: 'user',
        },
      ],
      model: MODEL,
    }),
    headers: { 'content-type': JSON_CONTENT_TYPE },
    method: 'POST',
    signal: overrides.signal,
  });

import { vi } from 'vitest';

export const API_BASE_URL = 'https://api:8880';
export const CONVERSATION_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d21';
export const JSON_CONTENT_TYPE = 'application/json';
export const MODEL = 'claude-sonnet-4-6';
export const RESPONSE_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d22';
export const USER_ID = 'anon-user-1';
export const OTHER_USER_ID = 'anon-user-2';

export type LoadedConversation = {
  readonly conversation: {
    readonly active_response_id: string | null;
    readonly active_status: string | null;
    readonly active_stream_id: string | null;
    readonly id: string;
    readonly user_id: string;
  };
  readonly messages: readonly unknown[];
};

type StateClientInput = {
  readonly activeResponseId?: string;
  readonly activeStreamId?: string;
  readonly content?: string;
  readonly conversationId: string;
  readonly messageId?: string;
  readonly metadata?: Record<string, unknown>;
  readonly model?: string;
  readonly responseId?: string;
  readonly streamId?: string;
  readonly userId: string;
};

type ResumeExistingStream = (
  streamId: string,
) => Promise<ReadableStream<Uint8Array> | null | undefined>;

const readStringStream = async (stream: ReadableStream<string>): Promise<string> => {
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

export const routeMocks = {
  activeChatProducers: {
    abort: vi.fn(),
    register: vi.fn(),
    unregister: vi.fn(),
  },
  consumedResumableStreams: [] as string[],
  createChatResumableStreamContext: vi.fn(),
  createChatStateClient: vi.fn(),
  resumableContext: {
    createNewResumableStream: vi.fn(
      async (streamId: string, makeStream: () => ReadableStream<string>) => {
        const stream = makeStream();
        routeMocks.consumedResumableStreams.push(await readStringStream(stream));
        void streamId;

        return stream;
      },
    ),
    resumeExistingStream: vi.fn<ResumeExistingStream>(async (_streamId) =>
      sseBody('data: resumed\n\n'),
    ),
  },
  stateClient: {
    clearActiveStreamIfCurrent: vi.fn(async (_input: StateClientInput) => undefined),
    loadConversation: vi.fn(async (_input: StateClientInput): Promise<LoadedConversation> => ({
      conversation: {
        active_response_id: RESPONSE_ID,
        active_status: 'streaming',
        active_stream_id: RESPONSE_ID,
        id: CONVERSATION_ID,
        user_id: USER_ID,
      },
      messages: [],
    })),
    setActiveStream: vi.fn(async (_input: StateClientInput) => undefined),
    stopActiveStreamIfCurrent: vi.fn(async (_input: StateClientInput) => undefined),
    upsertAssistantMessage: vi.fn(async (_input: StateClientInput) => undefined),
    upsertConversation: vi.fn(async (_input: StateClientInput) => undefined),
    upsertUserMessage: vi.fn(async (_input: StateClientInput) => undefined),
  },
};

export const resetRouteMocks = (): void => {
  vi.clearAllMocks();
  routeMocks.consumedResumableStreams.length = 0;
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
  vi.doMock('@/lib/chat-state-client', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@/lib/chat-state-client')>();

    return {
      ...actual,
      createChatStateClient: routeMocks.createChatStateClient,
    };
  });
  vi.doMock('@/lib/resumable-stream-context', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@/lib/resumable-stream-context')>();

    return {
      ...actual,
      activeChatProducers: routeMocks.activeChatProducers,
      createChatResumableStreamContext: routeMocks.createChatResumableStreamContext,
    };
  });
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
      userId: USER_ID,
    }),
    headers: { 'content-type': JSON_CONTENT_TYPE },
    method: 'POST',
    signal: overrides.signal,
  });

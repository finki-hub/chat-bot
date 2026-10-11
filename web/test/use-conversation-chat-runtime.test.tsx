import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useConversationChatRuntime } from '@/lib/use-conversation-chat-runtime';

const SPONSORED_MODEL_ID = 'gpt-6-luna';

type MessageUpdate =
  ((messages: MyUIMessage[]) => MyUIMessage[]) | MyUIMessage[];

type RuntimeDataPart = {
  readonly data: unknown;
  readonly type: `data-${string}`;
};

type RuntimeFinish = {
  readonly isAbort: boolean;
  readonly isError: boolean;
  readonly message: MyUIMessage;
};

type RuntimeOptions = {
  readonly onData?: (part: RuntimeDataPart) => void;
  readonly onError?: () => void;
  readonly onFinish?: (result: RuntimeFinish) => void;
};

const previousMessages: MyUIMessage[] = [
  {
    id: 'u1',
    parts: [{ text: 'Претходно прашање', type: 'text' }],
    role: 'user',
  },
  {
    id: 'a1',
    parts: [{ text: 'Претходен одговор', type: 'text' }],
    role: 'assistant',
  },
];

const useChatOptions: RuntimeOptions[] = [];
const setMessages = vi.fn<(messages: MessageUpdate) => void>();
const refreshConversations = vi.fn<() => Promise<void>>();
const getChatStatus = vi.fn<() => 'ready' | 'submitted'>(() => 'ready');
const regenerateMessage =
  vi.fn<
    (options: {
      messageId: string;
      metadata?: Record<string, unknown>;
    }) => Promise<void>
  >();
const { getChatExtras, refetchModels } = vi.hoisted(() => ({
  getChatExtras: vi.fn<() => { model: string; reasoning: boolean }>(),
  refetchModels: vi.fn<() => Promise<void>>(),
}));

vi.mock('@ai-sdk/react', () => ({
  useChat: (options: RuntimeOptions) => {
    useChatOptions.push(options);
    return {
      messages: previousMessages,
      regenerate: regenerateMessage,
      sendMessage: vi.fn<() => Promise<void>>(),
      setMessages,
      status: getChatStatus(),
      stop: vi.fn<() => void>(),
    };
  },
}));

vi.mock('@/lib/transport', () => ({
  buildChatTransport: vi.fn<
    (getExtras: () => { model: string; reasoning: boolean }) => {
      readonly kind: 'transport';
    }
  >((getExtras) => {
    getChatExtras.mockImplementation(getExtras);
    return { kind: 'transport' };
  }),
}));

vi.mock('@/lib/use-models', () => ({
  useModels: () => ({
    models: [{ availability: 'sponsored', id: SPONSORED_MODEL_ID }],
    refetch: refetchModels,
  }),
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: vi.fn<
    () => {
      readonly hydratingConversation: boolean;
      readonly retryHydration: () => void;
    }
  >(() => ({
    hydratingConversation: false,
    retryHydration: vi.fn<() => void>(),
  })),
}));

describe('useConversationChatRuntime sponsored errors', () => {
  beforeEach(() => {
    refreshConversations.mockReset();
    refreshConversations.mockResolvedValue();
    refetchModels.mockReset();
    refetchModels.mockResolvedValue();
    setMessages.mockReset();
    getChatStatus.mockReset();
    getChatStatus.mockReturnValue('ready');
    regenerateMessage.mockReset();
    regenerateMessage.mockResolvedValue();
    getChatExtras.mockReset();
    useChatOptions.length = 0;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('regenerates legacy history with the replacement model and retains provenance', async () => {
    const oldHistoryAnswer: MyUIMessage = {
      id: 'legacy-answer',
      metadata: { inferenceModel: 'claude-sonnet-5' },
      parts: [{ text: 'Historic answer', type: 'text' }],
      role: 'assistant',
    };
    previousMessages.push(oldHistoryAnswer);
    const { result } = renderHook(() =>
      useConversationChatRuntime({
        activeId: 'legacy-model-conversation',
        model: 'claude-sonnet-5',
        preserveEmptyHydrationIdRef: { current: null },
        reasoning: false,
        refreshConversations,
        setActiveId: vi.fn<(id: null | string) => void>(),
      }),
    );

    await act(async () => {
      await result.current.regenerate({ messageId: oldHistoryAnswer.id });
    });

    expect(regenerateMessage).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ messageId: oldHistoryAnswer.id }),
    );
    expect(getChatExtras().model).toBe('claude-sonnet-5-5');
    expect(result.current.messages.at(-1)?.metadata?.inferenceModel).toBe(
      'claude-sonnet-5',
    );

    previousMessages.pop();
  });

  it('refreshes the model catalog and preserves the prior conversation on terminal quota errors', async () => {
    const { result } = renderHook(() =>
      useConversationChatRuntime({
        activeId: 'conversation-1',
        model: SPONSORED_MODEL_ID,
        preserveEmptyHydrationIdRef: { current: null },
        reasoning: false,
        refreshConversations,
        setActiveId: vi.fn<(id: null | string) => void>(),
      }),
    );

    const options = useChatOptions.at(-1);
    if (options?.onData === undefined || options.onFinish === undefined) {
      throw new Error('Chat runtime callbacks were not registered');
    }
    const finishedMessage = previousMessages.find(
      (message) => message.id === 'a1',
    );
    if (finishedMessage === undefined) {
      throw new Error('Finished message fixture is missing');
    }

    act(() => {
      options.onData?.({
        data: {
          code: 'free_quota_exhausted',
          message: 'unsafe provider detail',
          // eslint-disable-next-line camelcase -- mirrors the SSE wire contract.
          resets_at: '2026-07-18T12:00:00Z',
        },
        type: 'data-error',
      });
      options.onFinish?.({
        isAbort: false,
        isError: true,
        message: finishedMessage,
      });
    });

    await waitFor(() => {
      expect(refetchModels).toHaveBeenCalledOnce();
    });

    expect(result.current.activeError).toStrictEqual({
      code: 'free_quota_exhausted',
      message: 'unsafe provider detail',
      // eslint-disable-next-line camelcase -- mirrors the SSE wire contract.
      resets_at: '2026-07-18T12:00:00Z',
    });

    expect(result.current.messages).toStrictEqual(previousMessages);
    expect(setMessages).not.toHaveBeenCalled();
  });

  it('keeps ordinary error handling from refreshing the sponsored catalog', async () => {
    renderHook(() =>
      useConversationChatRuntime({
        activeId: 'conversation-1',
        model: SPONSORED_MODEL_ID,
        preserveEmptyHydrationIdRef: { current: null },
        reasoning: false,
        refreshConversations,
        setActiveId: vi.fn<(id: null | string) => void>(),
      }),
    );

    const options = useChatOptions.at(-1);
    if (options?.onData === undefined) {
      throw new Error('Chat runtime data callback was not registered');
    }

    act(() => {
      options.onData?.({
        data: { code: 'agent_error', message: 'ordinary error' },
        type: 'data-error',
      });
    });

    await Promise.resolve();

    expect(refetchModels).not.toHaveBeenCalled();
  });

  it('ignores malformed streamed data parts', () => {
    const { result } = renderHook(() =>
      useConversationChatRuntime({
        activeId: 'conversation-1',
        model: SPONSORED_MODEL_ID,
        preserveEmptyHydrationIdRef: { current: null },
        reasoning: false,
        refreshConversations,
        setActiveId: vi.fn<(id: null | string) => void>(),
      }),
    );
    const options = useChatOptions.at(-1);
    if (options?.onData === undefined) {
      throw new Error('Chat runtime data callback was not registered');
    }

    act(() => {
      options.onData?.({ data: 'invalid error', type: 'data-error' });
      options.onData?.({
        data: { stage: 'missing label' },
        type: 'data-status',
      });
      options.onData?.({ data: {}, type: 'data-unknown' });
    });

    expect(result.current.activeError).toBeUndefined();
    expect(result.current.activeStatus).toBeUndefined();
  });

  it('finalizes client timing when a regenerated message keeps its target id', () => {
    getChatStatus.mockReturnValue('submitted');
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000);
    renderHook(() =>
      useConversationChatRuntime({
        activeId: 'conversation-1',
        model: SPONSORED_MODEL_ID,
        preserveEmptyHydrationIdRef: { current: null },
        reasoning: false,
        refreshConversations,
        setActiveId: vi.fn<(id: null | string) => void>(),
      }),
    );
    const options = useChatOptions.at(-1);
    if (options?.onFinish === undefined) {
      throw new Error('Chat runtime finish callback was not registered');
    }
    nowSpy.mockReturnValue(1_800);

    act(() => {
      options.onFinish?.({
        isAbort: false,
        isError: false,
        message: {
          id: 'a1',
          metadata: {
            replacementMessageId: 'a1',
            responseId: 'response-2',
          },
          parts: [{ text: 'Регенериран одговор', type: 'text' }],
          role: 'assistant',
        },
      });
    });

    const update = setMessages.mock.calls[0]?.[0];
    if (typeof update !== 'function') {
      throw new TypeError('Expected a functional message update');
    }
    const updated = update(previousMessages);

    expect(updated.at(1)?.metadata?.timing).toStrictEqual({
      totalMs: 800,
      ttftMs: 0,
    });
  });
});

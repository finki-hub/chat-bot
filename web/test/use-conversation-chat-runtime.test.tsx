import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { useConversationChatRuntime } from '@/lib/use-conversation-chat-runtime';

type RuntimeDataPart =
  | { readonly data: ErrorNotice; readonly type: 'data-error' }
  | { readonly data: Record<string, never>; readonly type: 'data-reset' }
  | { readonly data: { label: string }; readonly type: 'data-status' };

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
const setMessages = vi.fn<(messages: MyUIMessage[]) => void>();
const refreshConversations = vi.fn<() => Promise<void>>();
const { refetchModels } = vi.hoisted(() => ({
  refetchModels: vi.fn<() => Promise<void>>(),
}));

vi.mock('@ai-sdk/react', () => ({
  useChat: (options: RuntimeOptions) => {
    useChatOptions.push(options);
    return {
      messages: previousMessages,
      regenerate: vi.fn<() => Promise<void>>(),
      sendMessage: vi.fn<() => Promise<void>>(),
      setMessages,
      status: 'ready',
      stop: vi.fn<() => void>(),
    };
  },
}));

vi.mock('@/lib/transport', () => ({
  buildChatTransport: vi.fn<() => { readonly kind: 'transport' }>(() => ({
    kind: 'transport',
  })),
}));

vi.mock('@/lib/use-models', () => ({
  useModels: () => ({
    models: [{ availability: 'sponsored', id: 'gpt-5.6-luna' }],
    refetch: refetchModels,
  }),
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: vi.fn<() => boolean>(() => false),
}));

describe('useConversationChatRuntime sponsored errors', () => {
  beforeEach(() => {
    refreshConversations.mockReset();
    refreshConversations.mockResolvedValue();
    refetchModels.mockReset();
    refetchModels.mockResolvedValue();
    setMessages.mockReset();
    useChatOptions.length = 0;
  });

  it('refreshes the model catalog and preserves the prior conversation on terminal quota errors', async () => {
    const { result } = renderHook(() =>
      useConversationChatRuntime({
        activeId: 'conversation-1',
        model: 'gpt-5.6-luna',
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
        model: 'gpt-5.6-luna',
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
});

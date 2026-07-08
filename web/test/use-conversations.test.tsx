import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';

type UseChatOptions = {
  readonly id?: string;
  readonly resume?: boolean;
};

const chatState = { status: 'ready' };

const localStop = vi.fn<() => Promise<void> | void>();
const deleteChatConversation = vi.fn<(id: string) => Promise<void>>();
const stopChatStream =
  vi.fn<(id: string, snapshot?: unknown) => Promise<void>>();
const chatMessages: MyUIMessage[] = [];
const useChatOptions: UseChatOptions[] = [];

vi.mock('@ai-sdk/react', () => ({
  useChat: (options: UseChatOptions) => {
    useChatOptions.push(options);

    return {
      messages: chatMessages,
      regenerate: vi.fn<() => Promise<void>>(),
      sendMessage: vi.fn<(message: MyUIMessage) => Promise<void>>(),
      setMessages: vi.fn<(messages: MyUIMessage[]) => void>(),
      status: chatState.status,
      stop: localStop,
    };
  },
}));

vi.mock('posthog-js', () => ({
  posthog: { capture: vi.fn<(event: string) => void>() },
}));

vi.mock('@/lib/transport', () => ({
  buildChatTransport: vi.fn<() => { readonly kind: 'transport' }>(() => ({
    kind: 'transport',
  })),
  deleteChatConversation: (id: string) => deleteChatConversation(id),
  stopChatStream: (id: string, snapshot?: unknown) =>
    stopChatStream(id, snapshot),
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: vi.fn<() => void>(),
}));

vi.mock('@/lib/use-conversation-list', () => ({
  useConversationList: () => ({
    conversations: [],
    refreshConversations: vi.fn<() => Promise<void>>().mockResolvedValue(),
  }),
}));

vi.mock('@/lib/db', () => ({
  clearAllConversations: vi.fn<() => Promise<void>>().mockResolvedValue(),
  createConversation: vi.fn<() => Promise<{ id: string }>>().mockResolvedValue({
    id: 'created-conversation',
  }),
  deleteConversation: vi.fn<() => Promise<void>>().mockResolvedValue(),
  loadMessages: vi.fn<() => Promise<[]>>().mockResolvedValue([]),
  renameConversation: vi.fn<() => Promise<void>>().mockResolvedValue(),
  renameConversationIfTitle: vi
    .fn<() => Promise<boolean>>()
    .mockResolvedValue(true),
  replaceConversationMessages: vi.fn<() => Promise<void>>().mockResolvedValue(),
  saveMessages: vi.fn<() => Promise<void>>().mockResolvedValue(),
  setMessageFeedback: vi.fn<() => Promise<void>>().mockResolvedValue(),
}));

describe('useConversations resumable streaming', () => {
  beforeEach(() => {
    deleteChatConversation.mockReset();
    deleteChatConversation.mockResolvedValue();
    localStop.mockClear();
    chatMessages.length = 0;
    stopChatStream.mockReset();
    stopChatStream.mockResolvedValue();
    chatState.status = 'ready';
    useChatOptions.length = 0;
    useUiStore.setState({ activeConversationId: null, model: 'model-a' });
  });

  it('configures useChat with stable active conversation id and resume enabled only when active', () => {
    const { rerender } = renderHook(() => useConversations('model-a'));

    expect(useChatOptions.at(-1)).toMatchObject({ resume: false });
    expect(useChatOptions.at(-1)?.id).toBeUndefined();

    act(() => {
      useUiStore.setState({ activeConversationId: 'conv-active' });
    });
    rerender();

    expect(useChatOptions.at(-1)).toMatchObject({
      id: 'conv-active',
      resume: true,
    });
  });

  it('stops the active stream before selecting another conversation', async () => {
    useUiStore.setState({ activeConversationId: 'conv-a' });
    chatState.status = 'streaming';
    const calls: string[] = [];
    stopChatStream.mockImplementation(() => {
      calls.push('server');
      return Promise.resolve();
    });
    localStop.mockImplementation(() => {
      calls.push('local');
    });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      result.current.onSelect('conv-b');
    });

    expect(localStop).toHaveBeenCalledOnce();
    expect(useUiStore.getState().activeConversationId).toBe('conv-a');

    await waitFor(() => {
      expect(stopChatStream).toHaveBeenCalledWith('conv-a', undefined);
    });

    await waitFor(() => {
      expect(useUiStore.getState().activeConversationId).toBe('conv-b');
    });

    expect(calls).toStrictEqual(['local', 'server']);
  });

  it('stops the local chat before awaiting the BFF stop endpoint on explicit stop', async () => {
    useUiStore.setState({ activeConversationId: 'conv-stop' });
    const calls: string[] = [];
    let resolveServerStop: (() => void) | undefined;
    const serverStopReleased = new Promise<void>((resolve) => {
      resolveServerStop = resolve;
    });
    stopChatStream.mockImplementation(async () => {
      calls.push('server');
      await serverStopReleased;
    });
    localStop.mockImplementation(() => {
      calls.push('local');
    });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      result.current.onStop();
    });

    expect(calls).toStrictEqual(['local', 'server']);
    expect(stopChatStream).toHaveBeenCalledWith('conv-stop', undefined);

    if (resolveServerStop === undefined) {
      throw new Error('Server stop resolver was not captured');
    }
    resolveServerStop();
    await serverStopReleased;
  });

  it('sends only the active stream id when explicitly stopping a stream', async () => {
    useUiStore.setState({ activeConversationId: 'conv-stop' });
    chatMessages.push(
      {
        id: 'u1',
        parts: [{ text: 'Question', type: 'text' }],
        role: 'user',
      },
      {
        id: 'a1',
        metadata: {
          inferenceModel: 'model-a',
          responseId: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d22',
        },
        parts: [{ text: 'Partial answer', type: 'text' }],
        role: 'assistant',
      },
    );
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      result.current.onStop();
    });

    await waitFor(() => {
      expect(stopChatStream).toHaveBeenCalledWith('conv-stop', {
        activeStreamId: '018f0f36-2b1d-7cc0-a50b-5f2d90c91d22',
      });
    });
  });

  it('deletes server state before removing the local conversation', async () => {
    const { deleteConversation } = await import('@/lib/db');
    const calls: string[] = [];
    deleteChatConversation.mockImplementation(() => {
      calls.push('server');
      return Promise.resolve();
    });
    vi.mocked(deleteConversation).mockImplementation(() => {
      calls.push('local');
      return Promise.resolve();
    });
    const { result } = renderHook(() => useConversations('model-a'));

    await act(async () => {
      await result.current.onDelete('conv-delete');
    });

    expect(deleteChatConversation).toHaveBeenCalledWith('conv-delete');
    expect(deleteConversation).toHaveBeenCalledWith('conv-delete');
    expect(calls).toStrictEqual(['server', 'local']);
  });
});

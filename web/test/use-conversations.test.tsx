import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';

type UseChatOptions = {
  readonly id?: string;
  readonly resume?: boolean;
};

const {
  ChatConversationRequestError,
  DeleteChatConversationError,
  StopChatStreamError,
} = vi.hoisted(() => ({
  ChatConversationRequestError: class MockChatConversationRequestError extends Error {
    readonly status: number;

    constructor(status: number) {
      super('Chat conversation request failed');
      this.name = 'ChatConversationRequestError';
      this.status = status;
    }
  },
  DeleteChatConversationError: class MockDeleteChatConversationError extends Error {
    readonly status: number;

    constructor(status: number) {
      super('Delete chat conversation failed');
      this.name = 'DeleteChatConversationError';
      this.status = status;
    }
  },
  StopChatStreamError: class MockStopChatStreamError extends Error {
    readonly status: number;

    constructor(status: number) {
      super('Stop chat stream failed');
      this.name = 'StopChatStreamError';
      this.status = status;
    }
  },
}));

const chatState = { status: 'ready' };

const clearChatConversations = vi.fn<() => Promise<void>>();
const localStop = vi.fn<() => Promise<void> | void>();
const deleteChatConversation = vi.fn<(id: string) => Promise<void>>();
const refreshConversations = vi.fn<() => Promise<void>>();
const saveChatConversation = vi.fn<() => Promise<void>>();
const sendMessage = vi.fn<(message: MyUIMessage) => Promise<void>>();
const stopChatStream =
  vi.fn<(id: string, snapshot?: unknown) => Promise<void>>();
const chatMessages: MyUIMessage[] = [];
const ACTIVE_CONVERSATION_ID = 'conv-active';
const useChatOptions: UseChatOptions[] = [];

vi.mock('@ai-sdk/react', () => ({
  useChat: (options: UseChatOptions) => {
    useChatOptions.push(options);

    return {
      messages: chatMessages,
      regenerate: vi.fn<() => Promise<void>>(),
      sendMessage,
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
  ChatConversationRequestError,
  clearChatConversations: () => clearChatConversations(),
  deleteChatConversation: (id: string) => deleteChatConversation(id),
  DeleteChatConversationError,
  saveChatConversation: () => saveChatConversation(),
  stopChatStream: (id: string, snapshot?: unknown) =>
    stopChatStream(id, snapshot),
  StopChatStreamError,
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: vi.fn<() => void>(),
}));

vi.mock('@/lib/use-conversation-list', () => ({
  useConversationList: () => ({
    conversations: [],
    error: false,
    loading: false,
    refreshConversations,
  }),
}));

describe('useConversations resumable streaming', () => {
  beforeEach(() => {
    clearChatConversations.mockReset();
    clearChatConversations.mockResolvedValue();
    deleteChatConversation.mockReset();
    deleteChatConversation.mockResolvedValue();
    localStop.mockClear();
    refreshConversations.mockReset();
    refreshConversations.mockResolvedValue();
    saveChatConversation.mockReset();
    saveChatConversation.mockResolvedValue();
    sendMessage.mockReset();
    sendMessage.mockResolvedValue();
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
      useUiStore.setState({ activeConversationId: ACTIVE_CONVERSATION_ID });
    });
    rerender();

    expect(useChatOptions.at(-1)).toMatchObject({
      id: ACTIVE_CONVERSATION_ID,
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

  it('returns false and does not send when initial conversation creation fails', async () => {
    saveChatConversation.mockRejectedValueOnce(
      new ChatConversationRequestError(503),
    );
    const { result } = renderHook(() => useConversations('model-a'));
    let accepted = true;

    await act(async () => {
      accepted = await result.current.submitMessage('Прашање');
    });

    expect(accepted).toBe(false);
    expect(sendMessage).not.toHaveBeenCalled();
    expect(result.current.activeError).toStrictEqual({
      code: 'conversation_create',
      message:
        'Разговорот не можеше да се започне. Прашањето е зачувано за повторен обид.',
    });
  });

  it('returns false and preserves the draft error when conversation creation cannot reach the server', async () => {
    saveChatConversation.mockRejectedValueOnce(
      new TypeError('Failed to fetch'),
    );
    const { result } = renderHook(() => useConversations('model-a'));
    let accepted = true;

    await act(async () => {
      accepted = await result.current.submitMessage('Прашање');
    });

    expect(accepted).toBe(false);
    expect(sendMessage).not.toHaveBeenCalled();
    expect(result.current.activeError).toStrictEqual({
      code: 'conversation_create',
      message:
        'Разговорот не можеше да се започне. Прашањето е зачувано за повторен обид.',
    });
  });

  it('sends the first message without waiting for sidebar refresh', async () => {
    refreshConversations.mockRejectedValueOnce(new Error('list unavailable'));
    const { result } = renderHook(() => useConversations('model-a'));

    await act(async () => {
      await result.current.submitMessage('Прашање');
    });

    expect(sendMessage).toHaveBeenCalledOnce();
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

  it('deletes server state for a conversation', async () => {
    const calls: string[] = [];
    deleteChatConversation.mockImplementation(() => {
      calls.push('server');
      return Promise.resolve();
    });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      void result.current.onDelete('conv-delete');
    });

    await waitFor(() => {
      expect(refreshConversations).toHaveBeenCalled();
    });

    expect(deleteChatConversation).toHaveBeenCalledWith('conv-delete');
    expect(calls).toStrictEqual(['server']);
  });

  it('stops the active stream before deleting the active conversation', async () => {
    const activeConversationId = ACTIVE_CONVERSATION_ID;
    useUiStore.setState({ activeConversationId });
    chatState.status = 'streaming';
    const calls: string[] = [];
    stopChatStream.mockImplementation(() => {
      calls.push('stop-server');
      return Promise.resolve();
    });
    localStop.mockImplementation(() => {
      calls.push('stop-local');
    });
    deleteChatConversation.mockImplementation(() => {
      calls.push('delete-server');
      return Promise.resolve();
    });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      void result.current.onDelete(activeConversationId);
    });

    await waitFor(() => {
      expect(refreshConversations).toHaveBeenCalled();
    });

    expect(calls).toStrictEqual(['stop-local', 'stop-server', 'delete-server']);
  });

  it('returns false when stopping the active stream fails before deletion', async () => {
    useUiStore.setState({ activeConversationId: ACTIVE_CONVERSATION_ID });
    chatState.status = 'streaming';
    stopChatStream.mockRejectedValueOnce(new StopChatStreamError(503));
    const { result } = renderHook(() => useConversations('model-a'));
    let deleted = true;

    await act(async () => {
      deleted = await result.current.onDelete(ACTIVE_CONVERSATION_ID);
    });

    expect(deleted).toBe(false);
    expect(deleteChatConversation).not.toHaveBeenCalled();
  });

  it('stops the active stream before clearing all conversations', async () => {
    useUiStore.setState({ activeConversationId: ACTIVE_CONVERSATION_ID });
    chatState.status = 'streaming';
    const calls: string[] = [];
    stopChatStream.mockImplementation(() => {
      calls.push('stop-server');
      return Promise.resolve();
    });
    localStop.mockImplementation(() => {
      calls.push('stop-local');
    });
    clearChatConversations.mockImplementation(() => {
      calls.push('clear-server');
      return Promise.resolve();
    });
    const { result } = renderHook(() => useConversations('model-a'));

    await act(async () => {
      await result.current.onClearAll();
    });

    expect(calls).toStrictEqual(['stop-local', 'stop-server', 'clear-server']);
    expect(useUiStore.getState().activeConversationId).toBeNull();
  });

  it('returns false when stopping the active stream fails before clearing', async () => {
    useUiStore.setState({ activeConversationId: ACTIVE_CONVERSATION_ID });
    chatState.status = 'streaming';
    stopChatStream.mockRejectedValueOnce(new StopChatStreamError(503));
    const { result } = renderHook(() => useConversations('model-a'));
    let cleared = true;

    await act(async () => {
      cleared = await result.current.onClearAll();
    });

    expect(cleared).toBe(false);
    expect(clearChatConversations).not.toHaveBeenCalled();
  });

  it('keeps a newly selected conversation when active delete finishes later', async () => {
    const activeConversationId = ACTIVE_CONVERSATION_ID;
    useUiStore.setState({ activeConversationId });
    let resolveServerDelete: (() => void) | undefined;
    const serverDeleteReleased = new Promise<void>((resolve) => {
      resolveServerDelete = resolve;
    });
    deleteChatConversation.mockImplementation(async () => {
      await serverDeleteReleased;
    });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      void result.current.onDelete(activeConversationId);
    });

    await waitFor(() => {
      expect(deleteChatConversation).toHaveBeenCalledWith(activeConversationId);
    });

    act(() => {
      result.current.onSelect('conv-next');
    });

    await waitFor(() => {
      expect(useUiStore.getState().activeConversationId).toBe('conv-next');
    });

    if (resolveServerDelete === undefined) {
      throw new Error('Server delete resolver was not captured');
    }
    resolveServerDelete();

    await waitFor(() => {
      expect(refreshConversations).toHaveBeenCalled();
    });

    expect(useUiStore.getState().activeConversationId).toBe('conv-next');
  });

  it('removes the local conversation when the server conversation is already gone', async () => {
    deleteChatConversation.mockRejectedValueOnce(
      new DeleteChatConversationError(404),
    );
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      void result.current.onDelete('conv-local-only');
    });

    await waitFor(() => {
      expect(refreshConversations).toHaveBeenCalled();
    });
  });
});

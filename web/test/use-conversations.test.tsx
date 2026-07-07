import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';

type UseChatOptions = {
  readonly id?: string;
  readonly resume?: boolean;
};

const localStop = vi.fn<() => void>();
const stopChatStream = vi.fn<(id: string) => Promise<void>>();
const useChatOptions: UseChatOptions[] = [];

vi.mock('@ai-sdk/react', () => ({
  useChat: (options: UseChatOptions) => {
    useChatOptions.push(options);

    return {
      messages: [] as MyUIMessage[],
      regenerate: vi.fn(),
      sendMessage: vi.fn(),
      setMessages: vi.fn(),
      status: 'ready',
      stop: localStop,
    };
  },
}));

vi.mock('posthog-js', () => ({
  posthog: { capture: vi.fn() },
}));

vi.mock('@/lib/transport', () => ({
  buildChatTransport: vi.fn(() => ({ kind: 'transport' })),
  stopChatStream: (id: string) => stopChatStream(id),
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: vi.fn(),
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
  renameConversationIfTitle: vi.fn<() => Promise<boolean>>().mockResolvedValue(
    true,
  ),
  replaceConversationMessages: vi.fn<() => Promise<void>>().mockResolvedValue(),
  saveMessages: vi.fn<() => Promise<void>>().mockResolvedValue(),
  setMessageFeedback: vi.fn<() => Promise<void>>().mockResolvedValue(),
}));

describe('useConversations resumable streaming', () => {
  beforeEach(() => {
    localStop.mockClear();
    stopChatStream.mockReset();
    stopChatStream.mockResolvedValue();
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

  it('does not stop the server or local stream when selecting another conversation', () => {
    useUiStore.setState({ activeConversationId: 'conv-a' });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      result.current.onSelect('conv-b');
    });

    expect(stopChatStream).not.toHaveBeenCalled();
    expect(localStop).not.toHaveBeenCalled();
  });

  it('calls the BFF stop endpoint before stopping the local chat on explicit stop', async () => {
    useUiStore.setState({ activeConversationId: 'conv-stop' });
    const calls: string[] = [];
    stopChatStream.mockImplementation(async () => {
      calls.push('server');
    });
    localStop.mockImplementation(() => {
      calls.push('local');
    });
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      result.current.onStop();
    });

    await waitFor(() => {
      expect(calls).toStrictEqual(['server', 'local']);
    });
    expect(stopChatStream).toHaveBeenCalledWith('conv-stop');
  });
});

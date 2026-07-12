import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';

const chatState = { status: 'streaming' };
const localStop = vi.fn<() => Promise<void> | void>();
const stopChatStream =
  vi.fn<(id: string, snapshot?: unknown) => Promise<void>>();

vi.mock('@ai-sdk/react', () => ({
  useChat: () => ({
    messages: [] as MyUIMessage[],
    regenerate: () => Promise.resolve(),
    sendMessage: () => Promise.resolve(),
    setMessages: () => null,
    status: chatState.status,
    stop: localStop,
  }),
}));

vi.mock('posthog-js', () => ({
  posthog: { capture: () => null },
}));

vi.mock('@/lib/transport', () => ({
  buildChatTransport: () => ({}),
  deleteChatConversation: () => Promise.resolve(),
  saveChatConversation: () => Promise.resolve(),
  stopChatStream: (id: string, snapshot?: unknown) =>
    stopChatStream(id, snapshot),
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: () => false,
}));

vi.mock('@/lib/use-conversation-list', () => ({
  useConversationList: () => ({
    conversations: [],
    refreshConversations: () => Promise.resolve(),
  }),
}));

describe('conversation stop order', () => {
  beforeEach(() => {
    chatState.status = 'streaming';
    localStop.mockReset();
    stopChatStream.mockReset();
    stopChatStream.mockResolvedValue();
    useUiStore.setState({ activeConversationId: 'conv-a', model: 'model-a' });
  });

  it('waits for local stop before selecting another streaming conversation', async () => {
    let resolveLocalStop: (() => void) | undefined;
    const localStopReleased = new Promise<void>((resolve) => {
      resolveLocalStop = resolve;
    });
    localStop.mockReturnValue(localStopReleased);
    const { result } = renderHook(() => useConversations('model-a'));

    act(() => {
      result.current.onSelect('conv-b');
    });

    expect(useUiStore.getState().activeConversationId).toBe('conv-a');

    if (resolveLocalStop === undefined) {
      throw new Error('Local stop resolver was not captured');
    }
    resolveLocalStop();

    await waitFor(() => {
      expect(useUiStore.getState().activeConversationId).toBe('conv-b');
    });
  });
});

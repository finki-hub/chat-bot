import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';

type UseChatOptions = {
  readonly id?: string;
};

const previousMessage: MyUIMessage = {
  id: 'message-a',
  metadata: {},
  parts: [{ text: 'Conversation A', type: 'text' }],
  role: 'user',
};

vi.mock('@ai-sdk/react', () => ({
  useChat: (options: UseChatOptions) => ({
    messages: options.id === 'conversation-b' ? [] : [previousMessage],
    regenerate: () => Promise.resolve(),
    sendMessage: () => Promise.resolve(),
    setMessages: () => null,
    status: 'ready',
    stop: () => null,
  }),
}));

vi.mock('@/lib/transport', () => ({
  buildChatTransport: () => ({}),
  deleteChatConversation: () => Promise.resolve(),
  saveChatConversation: () => Promise.resolve(),
  stopChatStream: () => Promise.resolve(),
}));

vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: () => true,
}));

vi.mock('@/lib/use-conversation-list', () => ({
  useConversationList: () => ({
    conversations: [],
    refreshConversations: () => Promise.resolve(),
  }),
}));

const RuntimeSwitchHarness = () => {
  const { messages, onSelect, status } = useConversations('model-a');

  return (
    <>
      <button
        onClick={() => {
          onSelect('conversation-b');
        }}
        type="button"
      >
        Select conversation B
      </button>
      <Thread
        messages={messages}
        status={status}
      />
    </>
  );
};

describe('conversation switch runtime', () => {
  it('keeps previous messages visible while the selected chat runtime is empty and hydrating', () => {
    useUiStore.setState({ activeConversationId: 'conversation-a' });
    render(<RuntimeSwitchHarness />);

    fireEvent.click(
      screen.getByRole('button', { name: 'Select conversation B' }),
    );

    expect(screen.queryByText('Започни разговор')).not.toBeInTheDocument();
    expect(screen.getByText('Conversation A')).toBeInTheDocument();
  });
});

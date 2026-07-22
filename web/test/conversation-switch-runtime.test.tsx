import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

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

const freshAssistantMessage: MyUIMessage = {
  id: 'message-b',
  metadata: {},
  parts: [{ text: 'Fresh answer', type: 'text' }],
  role: 'assistant',
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

vi.mock('@/lib/use-models', () => ({
  useModels: () => ({
    models: [],
    refetch: () => Promise.resolve(),
  }),
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
  beforeEach(() => {
    useUiStore.setState({
      activeConversationId: null,
      model: 'model-a',
      reasoning: false,
      sidebarOpen: false,
    });
  });

  it('keeps previous messages visible without replaying their entrance animation while the selected chat hydrates', () => {
    useUiStore.setState({ activeConversationId: 'conversation-a' });
    render(<RuntimeSwitchHarness />);

    fireEvent.click(
      screen.getByRole('button', { name: 'Select conversation B' }),
    );

    expect(screen.queryByText('Започни разговор')).not.toBeInTheDocument();

    const previousMessageShell = screen
      .getByText('Conversation A')
      .closest('.group');

    expect(previousMessageShell).toBeInTheDocument();
    expect(previousMessageShell).not.toHaveClass('motion-safe:animate-in');
  });

  it('animates a newly submitted user message', () => {
    render(
      <Thread
        messages={[previousMessage]}
        status="submitted"
      />,
    );

    const newMessageShell = screen
      .getByText('Conversation A')
      .closest('.group');

    expect(newMessageShell).toHaveClass('motion-safe:animate-in');
  });

  it('animates a newly streaming assistant message', () => {
    render(
      <Thread
        messages={[previousMessage, freshAssistantMessage]}
        status="streaming"
      />,
    );

    const animatedShell = screen
      .getByText('Fresh answer')
      .closest('[class~="motion-safe:animate-in"]');

    expect(animatedShell).toBeInTheDocument();
  });
});

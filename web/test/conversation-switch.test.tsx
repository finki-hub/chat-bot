import { fireEvent, render, screen } from '@testing-library/react';
import { useRef, useState } from 'react';
import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { useConversationManagement } from '@/lib/use-conversation-management';

const previousMessage: MyUIMessage = {
  id: 'message-a',
  metadata: {},
  parts: [{ text: 'Conversation A', type: 'text' }],
  role: 'user',
};

const ConversationSwitchHarness = () => {
  const [messages, setMessages] = useState<MyUIMessage[]>([previousMessage]);
  const convoIdRef = useRef<null | string>('conversation-a');
  const preserveEmptyHydrationIdRef = useRef<null | string>(null);
  const { handleSelect } = useConversationManagement({
    applyGeneratedTitle: () => Promise.resolve(),
    convoIdRef,
    handleStop: () => Promise.resolve(),
    model: 'model-a',
    preserveEmptyHydrationIdRef,
    refreshConversations: () => Promise.resolve(),
    sendMessageRef: useRef(() => Promise.resolve()),
    setActiveError: () => null,
    setActiveId: () => null,
    setMessages,
    status: 'ready',
  });

  return (
    <>
      <button
        onClick={() => {
          handleSelect('conversation-b');
        }}
        type="button"
      >
        Select conversation B
      </button>
      <Thread
        messages={messages}
        status="ready"
      />
    </>
  );
};

describe('conversation switching', () => {
  it('keeps the current thread visible until the selected conversation hydrates', () => {
    render(<ConversationSwitchHarness />);

    fireEvent.click(
      screen.getByRole('button', { name: 'Select conversation B' }),
    );

    expect(screen.queryByText('Започни разговор')).not.toBeInTheDocument();
    expect(screen.getByText('Conversation A')).toBeInTheDocument();
  });
});

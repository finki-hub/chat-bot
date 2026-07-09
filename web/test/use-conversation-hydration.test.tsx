import { renderHook, waitFor } from '@testing-library/react';
import { useRef, useState } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { useConversationHydration } from '@/lib/use-conversation-hydration';

type ActiveStatus =
  | undefined
  | { readonly label: string; readonly tool?: string };

const transportMocks = vi.hoisted(() => ({
  loadChatConversationHistory:
    vi.fn<
      (
        conversationId: string,
      ) => Promise<null | { readonly messages: readonly MyUIMessage[] }>
    >(),
}));

vi.mock('@/lib/transport', () => transportMocks);

const message = (id: string, text: string): MyUIMessage => ({
  id,
  metadata: {},
  parts: [{ text, type: 'text' }],
  role: 'user',
});

const useHydratedMessages = (): MyUIMessage[] => {
  const [messages, setMessages] = useState<MyUIMessage[]>(() => [
    message('previous-a', 'Conversation A'),
  ]);
  const convoIdRef = useRef<null | string>(null);
  useConversationHydration({
    activeId: 'conversation-b',
    convoIdRef,
    setActiveError: vi.fn<(value: ErrorNotice | undefined) => void>(),
    setActiveId: vi.fn<(id: null | string) => void>(),
    setActiveStatus: vi.fn<(value: ActiveStatus) => void>(),
    setMessages,
  });

  return messages;
};

describe('useConversationHydration', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      messages: [message('server-b', 'Conversation B')],
    });
  });

  it('replaces previous conversation messages when hydrating a selected server conversation', async () => {
    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe('server-b');
    });

    expect(transportMocks.loadChatConversationHistory).toHaveBeenCalledWith(
      'conversation-b',
    );
  });
});

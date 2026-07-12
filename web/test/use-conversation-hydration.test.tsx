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

const useHydratedMessages = ({
  activeStreamConversationId,
  preserveEmptyHydrationId,
}: {
  readonly activeStreamConversationId?: string;
  readonly preserveEmptyHydrationId?: string;
} = {}): MyUIMessage[] => {
  const [messages, setMessages] = useState<MyUIMessage[]>(() => [
    message('previous-a', 'Conversation A'),
  ]);
  const convoIdRef = useRef<null | string>(null);
  const preserveEmptyHydrationIdRef = useRef<null | string>(
    preserveEmptyHydrationId ?? null,
  );
  const activeStreamConversationIdRef = useRef<null | string>(
    activeStreamConversationId ?? null,
  );
  const setActiveError = useRef(
    vi.fn<(value: ErrorNotice | undefined) => void>(),
  ).current;
  const setActiveId = useRef(vi.fn<(id: null | string) => void>()).current;
  const setActiveStatus = useRef(
    vi.fn<(value: ActiveStatus) => void>(),
  ).current;
  useConversationHydration({
    activeId: 'conversation-b',
    activeStreamConversationIdRef,
    convoIdRef,
    preserveEmptyHydrationIdRef,
    setActiveError,
    setActiveId,
    setActiveStatus,
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

  it('clears previous messages when the selected server conversation is empty', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      messages: [],
    });

    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(0);
    });
  });

  it('preserves local messages when a locally created conversation has not saved history yet', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      messages: [],
    });

    const { result } = renderHook(() =>
      useHydratedMessages({ preserveEmptyHydrationId: 'conversation-b' }),
    );

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe('previous-a');
    });
  });

  it('preserves resumed stream messages when server history is still empty', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      messages: [],
    });

    const { result } = renderHook(() =>
      useHydratedMessages({ activeStreamConversationId: 'conversation-b' }),
    );

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe('previous-a');
    });
  });

  it('clears previous messages when selected conversation history rejects', async () => {
    transportMocks.loadChatConversationHistory.mockRejectedValue(
      new Error('network failed'),
    );

    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(0);
    });
  });
});

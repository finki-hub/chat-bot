import { renderHook, waitFor } from '@testing-library/react';
import { useRef, useState } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { useConversationHydration } from '@/lib/use-conversation-hydration';

type ActiveStatus =
  | undefined
  | { readonly label: string; readonly tool?: string };

const ACTIVE_ID = 'conversation-b';
const PREVIOUS_ID = 'previous-a';
const PREVIOUS_TEXT = 'Conversation A';

const transportMocks = vi.hoisted(() => {
  class ChatConversationRequestError extends Error {
    readonly status: number;

    constructor(status: number, options?: ErrorOptions) {
      super('Chat conversation request failed', options);
      this.name = 'ChatConversationRequestError';
      this.status = status;
    }
  }

  return {
    ChatConversationRequestError,
    loadChatConversationHistory: vi.fn<
      (conversationId: string) => Promise<null | {
        readonly conversation: {
          readonly activeStream: null | {
            readonly id: string;
            readonly replacementMessageId: null | string;
          };
        };
        readonly messages: readonly MyUIMessage[];
      }>
    >(),
  };
});
const setActiveError = vi.fn<(value: ErrorNotice | undefined) => void>();
const setActiveId = vi.fn<(id: null | string) => void>();

vi.mock('@/lib/transport', () => transportMocks);

const message = (id: string, text: string): MyUIMessage => ({
  id,
  metadata: {},
  parts: [{ text, type: 'text' }],
  role: 'user',
});

const useHydratedMessages = ({
  activeStreamConversationId,
  initialMessages,
  preserveEmptyHydrationId,
}: {
  readonly activeStreamConversationId?: string;
  readonly initialMessages?: readonly MyUIMessage[];
  readonly preserveEmptyHydrationId?: string;
} = {}): MyUIMessage[] => {
  const [messages, setMessages] = useState<MyUIMessage[]>(() =>
    initialMessages === undefined
      ? [message(PREVIOUS_ID, PREVIOUS_TEXT)]
      : [...initialMessages],
  );
  const convoIdRef = useRef<null | string>(null);
  const preserveEmptyHydrationIdRef = useRef<null | string>(
    preserveEmptyHydrationId ?? null,
  );
  const activeStreamConversationIdRef = useRef<null | string>(
    activeStreamConversationId ?? null,
  );
  const setActiveStatus = useRef(
    vi.fn<(value: ActiveStatus) => void>(),
  ).current;
  useConversationHydration({
    activeId: ACTIVE_ID,
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
    setActiveError.mockReset();
    setActiveId.mockReset();
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      conversation: { activeStream: null },
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
      ACTIVE_ID,
    );
  });

  it('clears previous messages when the selected server conversation is empty', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      conversation: { activeStream: null },
      messages: [],
    });

    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(0);
    });
  });

  it('preserves local messages when a locally created conversation has not saved history yet', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      conversation: { activeStream: null },
      messages: [],
    });

    const { result } = renderHook(() =>
      useHydratedMessages({ preserveEmptyHydrationId: ACTIVE_ID }),
    );

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe(PREVIOUS_ID);
    });
  });

  it('preserves resumed stream messages when server history is still empty', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      conversation: { activeStream: null },
      messages: [],
    });

    const { result } = renderHook(() =>
      useHydratedMessages({ activeStreamConversationId: ACTIVE_ID }),
    );

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe(PREVIOUS_ID);
    });
  });

  it('merges non-empty persisted history with an already resumed assistant', async () => {
    const resumedAssistant: MyUIMessage = {
      id: 'stream-answer',
      metadata: { responseId: 'active-response' },
      parts: [{ text: 'Resumed answer', type: 'text' }],
      role: 'assistant',
    };
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      conversation: {
        activeStream: {
          id: 'active-response',
          replacementMessageId: null,
        },
      },
      messages: [message('server-user', 'Persisted question')],
    });

    const { result } = renderHook(() =>
      useHydratedMessages({ initialMessages: [resumedAssistant] }),
    );

    await waitFor(() => {
      expect(result.current.map((item) => item.id)).toStrictEqual([
        'server-user',
        'stream-answer',
      ]);
    });
  });

  it('prunes stale later turns when regeneration history arrives before the stream', async () => {
    transportMocks.loadChatConversationHistory.mockResolvedValue({
      conversation: {
        activeStream: {
          id: 'active-response',
          replacementMessageId: 'server-answer',
        },
      },
      messages: [
        message('server-user', 'Persisted question'),
        {
          id: 'server-answer',
          metadata: { responseId: 'old-response' },
          parts: [{ text: 'Old answer', type: 'text' }],
          role: 'assistant',
        },
        message('later-user', 'Stale follow-up'),
      ],
    });

    const { result } = renderHook(() =>
      useHydratedMessages({ initialMessages: [] }),
    );

    await waitFor(() => {
      expect(result.current.map((item) => item.id)).toStrictEqual([
        'server-user',
        'server-answer',
      ]);
    });

    expect(result.current.at(1)?.parts).toStrictEqual([]);
  });

  it('preserves previous messages and surfaces an error when selected conversation history rejects', async () => {
    transportMocks.loadChatConversationHistory.mockRejectedValue(
      new transportMocks.ChatConversationRequestError(503),
    );

    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe(PREVIOUS_ID);
    });

    expect(setActiveId).not.toHaveBeenCalledWith(null);
    expect(setActiveError).toHaveBeenCalledWith({
      code: 'history_load',
      message: 'Разговорот не можеше да се вчита. Обидете се повторно.',
    });
  });

  it('clears a selected conversation only when history reports it missing', async () => {
    const { ChatConversationRequestError } = await import('@/lib/transport');
    transportMocks.loadChatConversationHistory.mockRejectedValue(
      new ChatConversationRequestError(404),
    );

    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(0);
    });

    expect(setActiveId).toHaveBeenCalledWith(null);
  });

  it('preserves a locally created conversation when history is not available yet', async () => {
    transportMocks.loadChatConversationHistory.mockRejectedValue(
      new transportMocks.ChatConversationRequestError(404),
    );

    const { result } = renderHook(() =>
      useHydratedMessages({ preserveEmptyHydrationId: ACTIVE_ID }),
    );

    await waitFor(() => {
      expect(transportMocks.loadChatConversationHistory).toHaveBeenCalledWith(
        ACTIVE_ID,
      );
    });

    expect(result.current).toHaveLength(1);
    expect(result.current[0]?.id).toBe(PREVIOUS_ID);
    expect(setActiveId).not.toHaveBeenCalledWith(null);
  });
});

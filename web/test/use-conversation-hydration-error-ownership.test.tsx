import { act, renderHook, waitFor } from '@testing-library/react';
import { useRef, useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { useConversationHydration } from '@/lib/use-conversation-hydration';

const ACTIVE_ID = 'active-conversation';

const transportMocks = vi.hoisted(() => ({
  ChatConversationRequestError: class extends Error {
    readonly status = 500;
  },
  loadChatConversationHistory: vi.fn<
    (conversationId: string) => Promise<null | {
      readonly conversation: { readonly activeStream: null };
      readonly messages: readonly MyUIMessage[];
    }>
  >(),
}));

vi.mock('@/lib/transport', () => transportMocks);

const useHydrationErrorHarness = () => {
  const [activeError, setActiveError] = useState<ErrorNotice | undefined>();
  const [messages, setMessages] = useState<MyUIMessage[]>([]);
  const activeStreamConversationIdRef = useRef<null | string>(null);
  const convoIdRef = useRef<null | string>(null);
  const preserveEmptyHydrationIdRef = useRef<null | string>(null);
  const setActiveId = useRef(vi.fn<(id: null | string) => void>()).current;
  const setActiveStatus = useRef(
    vi.fn<(value: undefined | { label: string; tool?: string }) => void>(),
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

  return { activeError, messages, setActiveError };
};

describe('useConversationHydration error ownership', () => {
  it('preserves a newer stream error when deferred history succeeds', async () => {
    // Given history is pending when the active stream reports a newer error.
    const history = {
      conversation: { activeStream: null },
      messages: [
        {
          id: 'server-message',
          metadata: {},
          parts: [{ text: 'Persisted answer', type: 'text' }],
          role: 'assistant',
        },
      ],
    } satisfies {
      conversation: { activeStream: null };
      messages: MyUIMessage[];
    };
    let releaseHistory: ((value: typeof history) => void) | undefined;
    transportMocks.loadChatConversationHistory.mockReturnValueOnce(
      new Promise((resolve) => {
        releaseHistory = resolve;
      }),
    );
    const { result } = renderHook(useHydrationErrorHarness);
    const streamError = {
      code: 'network',
      message: 'stream failed after history started',
    };

    // When the stream error arrives before history completes.
    act(() => {
      result.current.setActiveError(streamError);
    });
    if (releaseHistory === undefined) {
      throw new Error('History release was not initialized');
    }
    const resolveHistory = releaseHistory;
    act(() => {
      resolveHistory(history);
    });

    // Then hydration cannot clear the newer stream failure.
    await waitFor(() => {
      expect(result.current.messages.at(0)?.id).toBe('server-message');
    });

    expect(result.current.activeError).toStrictEqual(streamError);
  });
});

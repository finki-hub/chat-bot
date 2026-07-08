import { renderHook, waitFor } from '@testing-library/react';
import { useRef, useState } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';
import type { ConversationRow, MessageRow } from '@/lib/db';

import { useConversationHydration } from '@/lib/use-conversation-hydration';

type ActiveStatus =
  | undefined
  | { readonly label: string; readonly tool?: string };

type CreateConversationInput = {
  readonly id?: string;
  readonly model: string;
  readonly title: string;
};

const dbMocks = vi.hoisted(() => ({
  createConversation:
    vi.fn<(input: CreateConversationInput) => Promise<ConversationRow>>(),
  getConversation:
    vi.fn<(id: string) => Promise<ConversationRow | undefined>>(),
  loadMessages: vi.fn<(conversationId: string) => Promise<MessageRow[]>>(),
  replaceConversationMessages:
    vi.fn<(conversationId: string, messages: MyUIMessage[]) => Promise<void>>(),
}));

vi.mock('@/lib/db', () => dbMocks);

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
    model: 'model-a',
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
    dbMocks.createConversation.mockResolvedValue({
      createdAt: 1,
      id: 'conversation-b',
      model: 'model-a',
      title: 'Conversation B',
      updatedAt: 1,
    });
    dbMocks.getConversation.mockResolvedValue({
      createdAt: 1,
      id: 'conversation-b',
      model: 'model-a',
      title: 'Conversation B',
      updatedAt: 1,
    });
    dbMocks.loadMessages.mockResolvedValue([
      {
        conversationId: 'conversation-b',
        createdAt: 1,
        id: 'local-b',
        metadata: {},
        parts: [{ text: 'Conversation B', type: 'text' }],
        role: 'user',
      },
    ]);
    dbMocks.replaceConversationMessages.mockResolvedValue(undefined);
    vi.stubGlobal(
      'fetch',
      vi.fn<() => Promise<Response>>().mockResolvedValue(new Response(null)),
    );
  });

  it('replaces previous conversation messages when hydrating a selected local conversation', async () => {
    const { result } = renderHook(useHydratedMessages);

    await waitFor(() => {
      expect(result.current).toHaveLength(1);
      expect(result.current[0]?.id).toBe('local-b');
    });
  });
});

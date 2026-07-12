import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/conversation-types';

import { useConversationList } from '@/lib/use-conversation-list';

const transportMocks = vi.hoisted(() => ({
  ChatConversationRequestError: class MockChatConversationRequestError extends Error {
    readonly status: number;

    constructor(status: number) {
      super('Chat conversation request failed');
      this.name = 'ChatConversationRequestError';
      this.status = status;
    }
  },
  listChatConversations: vi.fn<() => Promise<ConversationRow[]>>(),
}));

vi.mock('@/lib/transport', () => transportMocks);

describe('useConversationList', () => {
  beforeEach(() => {
    transportMocks.listChatConversations.mockReset();
  });

  it('surfaces a load error and retries without discarding the last list', async () => {
    const conversations = [{ id: 'conv-1', model: null, title: 'Запишување' }];
    transportMocks.listChatConversations
      .mockResolvedValueOnce(conversations)
      .mockRejectedValueOnce(new TypeError('network failed'))
      .mockResolvedValueOnce(conversations);
    const { result } = renderHook(useConversationList);

    await waitFor(() => {
      expect(result.current.conversations).toStrictEqual(conversations);
    });

    await act(async () => {
      await result.current.refreshConversations();
    });

    expect(result.current.conversations).toStrictEqual(conversations);
    expect(result.current.error).toBe(true);

    await act(async () => {
      await result.current.refreshConversations();
    });

    expect(result.current.error).toBe(false);
  });
});

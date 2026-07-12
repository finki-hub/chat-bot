import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useGeneratedTitle } from '@/lib/use-generated-title';

type GenerateChatTitle = (input: {
  readonly messages: readonly MyUIMessage[];
  readonly providerModel?: string;
}) => Promise<null | string>;

const generateChatTitle = vi.fn<GenerateChatTitle>();
const loadChatConversationHistory = vi.fn<() => Promise<null>>();
const refreshConversations = vi.fn<() => Promise<void>>();
const saveChatConversation = vi.fn<() => Promise<void>>();

vi.mock('@/lib/chat-title', () => ({
  generateChatTitle: (input: { readonly messages: readonly MyUIMessage[] }) =>
    generateChatTitle(input),
}));

vi.mock('@/lib/transport', () => ({
  loadChatConversationHistory: () => loadChatConversationHistory(),
  saveChatConversation: () => saveChatConversation(),
}));

beforeEach(() => {
  generateChatTitle.mockReset();
  generateChatTitle.mockResolvedValue('Јунска испитна сесија');
  refreshConversations.mockReset();
  refreshConversations.mockResolvedValue();
  saveChatConversation.mockReset();
  saveChatConversation.mockResolvedValue();
});

test('requests a generated title without coupling it to the active chat model', async () => {
  const messages: readonly MyUIMessage[] = [
    {
      id: 'user-1',
      parts: [{ text: 'Кога е јунската сесија?', type: 'text' }],
      role: 'user',
    },
  ];
  const options = {
    conversations: [],
    modelRef: { current: 'claude-sonnet-5' },
    refreshConversations,
  };
  const { result } = renderHook(() => useGeneratedTitle(options));

  await act(async () => {
    await result.current.applyGeneratedTitle(
      'conversation-1',
      messages,
      'Old title',
    );
  });

  expect(generateChatTitle).toHaveBeenCalledWith({
    messages,
    providerModel: 'claude-sonnet-5',
  });
  expect(saveChatConversation).toHaveBeenCalledOnce();
  expect(refreshConversations).toHaveBeenCalledOnce();
});

import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';
import type { ChatConversationHistory } from '@/lib/conversation-types';
import type { SaveChatConversationInput } from '@/lib/transport';

import { useGeneratedTitle } from '@/lib/use-generated-title';

type GenerateChatTitle = (input: {
  readonly messages: readonly MyUIMessage[];
  readonly providerModel?: string;
}) => Promise<null | string>;

const generateChatTitle = vi.fn<GenerateChatTitle>();
const loadChatConversationHistory =
  vi.fn<(conversationId: string) => Promise<ChatConversationHistory | null>>();
const refreshConversations = vi.fn<() => Promise<void>>();
const saveChatConversation =
  vi.fn<(input: SaveChatConversationInput) => Promise<void>>();

const CONVERSATION_ID = 'conversation-1';
const GENERATED_TITLE = 'Јунска испитна сесија';
const MODEL_ID = 'claude-sonnet-5';
const OLD_TITLE = 'Old title';

vi.mock('@/lib/chat-title', () => ({
  generateChatTitle: (input: { readonly messages: readonly MyUIMessage[] }) =>
    generateChatTitle(input),
}));

vi.mock('@/lib/transport', () => ({
  loadChatConversationHistory: (conversationId: string) =>
    loadChatConversationHistory(conversationId),
  saveChatConversation: (input: SaveChatConversationInput) =>
    saveChatConversation(input),
}));

beforeEach(() => {
  generateChatTitle.mockReset();
  generateChatTitle.mockResolvedValue(GENERATED_TITLE);
  loadChatConversationHistory.mockReset();
  loadChatConversationHistory.mockResolvedValue(null);
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
    modelRef: { current: MODEL_ID },
    refreshConversations,
  };
  const { result } = renderHook(() => useGeneratedTitle(options));

  await act(async () => {
    await result.current.applyGeneratedTitle(
      CONVERSATION_ID,
      messages,
      OLD_TITLE,
    );
  });

  expect(generateChatTitle).toHaveBeenCalledWith({
    messages,
    providerModel: MODEL_ID,
  });
  expect(saveChatConversation).toHaveBeenCalledOnce();
  expect(refreshConversations).toHaveBeenCalledOnce();
});

test('loads the requested conversation before generating its title', async () => {
  const messages: readonly MyUIMessage[] = [
    {
      id: 'user-1',
      parts: [{ text: 'Кога е јунската сесија?', type: 'text' }],
      role: 'user',
    },
  ];
  loadChatConversationHistory.mockResolvedValue({
    conversation: {
      id: CONVERSATION_ID,
      model: MODEL_ID,
      title: OLD_TITLE,
    },
    messages,
  });
  const { result } = renderHook(() =>
    useGeneratedTitle({
      conversations: [
        {
          id: CONVERSATION_ID,
          model: MODEL_ID,
          title: OLD_TITLE,
        },
      ],
      modelRef: { current: MODEL_ID },
      refreshConversations,
    }),
  );

  act(() => {
    result.current.handleGenerateTitle(CONVERSATION_ID);
  });

  await waitFor(() => {
    expect(loadChatConversationHistory).toHaveBeenCalledWith(CONVERSATION_ID);
  });
  await waitFor(() => {
    expect(saveChatConversation).toHaveBeenCalledWith({
      expectedTitle: OLD_TITLE,
      id: CONVERSATION_ID,
      title: GENERATED_TITLE,
    });
  });
});

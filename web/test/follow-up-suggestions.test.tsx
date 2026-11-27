import { render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { useFollowUpSuggestions } from '@/lib/use-follow-up-suggestions';

type FetchChatSuggestions = (input: {
  readonly answer: string;
  readonly conversationId: string;
  readonly providerModel?: string;
  readonly question: string;
}) => Promise<readonly string[]>;

const fetchChatSuggestions = vi.fn<FetchChatSuggestions>();

vi.mock('@/lib/chat-suggestions', () => ({
  fetchChatSuggestions: (input: Parameters<FetchChatSuggestions>[0]) =>
    fetchChatSuggestions(input),
}));

const QUESTION: MyUIMessage = {
  id: 'q1',
  parts: [{ text: 'Колку чини уписот?', type: 'text' }],
  role: 'user',
};
const ANSWER: MyUIMessage = {
  id: 'a1',
  metadata: { responseId: 'r1' },
  parts: [{ text: 'Уписот чини 200 денари.', type: 'text' }],
  role: 'assistant',
};
const SUGGESTIONS = ['Како се плаќа уписот?', 'Кој е рокот за упис?'];

type Status = 'error' | 'ready' | 'streaming' | 'submitted';

const renderFollowUps = (
  initialStatus: Status,
  messages = [QUESTION, ANSWER],
) =>
  renderHook(
    ({ status }: { status: Status }) =>
      useFollowUpSuggestions({
        activeError: undefined,
        conversationId: 'c1',
        messages,
        model: 'claude-sonnet-5-5',
        status,
      }),
    { initialProps: { status: initialStatus } },
  );

beforeEach(() => {
  fetchChatSuggestions.mockReset();
  fetchChatSuggestions.mockResolvedValue(SUGGESTIONS);
});

describe('useFollowUpSuggestions', () => {
  it('asks once an answer finishes streaming', async () => {
    const { rerender, result } = renderFollowUps('streaming');

    rerender({ status: 'ready' });

    await waitFor(() => {
      expect(result.current).toStrictEqual({
        messageId: 'a1',
        suggestions: SUGGESTIONS,
      });
    });

    expect(fetchChatSuggestions).toHaveBeenCalledExactlyOnceWith({
      answer: 'Уписот чини 200 денари.',
      conversationId: 'c1',
      providerModel: 'claude-sonnet-5-5',
      question: 'Колку чини уписот?',
    });
  });

  it('does not ask for an answer loaded from history', () => {
    renderFollowUps('ready');

    expect(fetchChatSuggestions).not.toHaveBeenCalled();
  });

  it('does not ask for a failed answer', () => {
    const failed: MyUIMessage = {
      ...ANSWER,
      metadata: { error: { code: 'agent_error', message: 'x' } },
    };
    const { rerender } = renderFollowUps('streaming', [QUESTION, failed]);

    rerender({ status: 'ready' });

    expect(fetchChatSuggestions).not.toHaveBeenCalled();
  });

  it('clears the suggestions when the next question is sent', async () => {
    const { rerender, result } = renderFollowUps('streaming');
    rerender({ status: 'ready' });
    await waitFor(() => {
      expect(result.current).not.toBeNull();
    });

    rerender({ status: 'submitted' });

    expect(result.current).toBeNull();
  });
});

describe('Thread follow-ups', () => {
  it('sends a picked suggestion as the next question', async () => {
    const onPick = vi.fn<(text: string) => void>();
    render(
      <Thread
        followUps={{ messageId: 'a1', suggestions: SUGGESTIONS }}
        messages={[QUESTION, ANSWER]}
        onPickSuggestion={onPick}
        status="ready"
      />,
    );

    await userEvent.click(
      screen.getByRole('button', { name: 'Кој е рокот за упис?' }),
    );

    expect(onPick).toHaveBeenCalledExactlyOnceWith('Кој е рокот за упис?');
  });

  it('hides suggestions made for another answer or while streaming', () => {
    const { rerender } = render(
      <Thread
        followUps={{ messageId: 'other', suggestions: SUGGESTIONS }}
        messages={[QUESTION, ANSWER]}
        onPickSuggestion={vi.fn<(text: string) => void>()}
        status="ready"
      />,
    );

    expect(
      screen.queryByRole('navigation', { name: 'Следни прашања' }),
    ).not.toBeInTheDocument();

    rerender(
      <Thread
        followUps={{ messageId: 'a1', suggestions: SUGGESTIONS }}
        messages={[QUESTION, ANSWER]}
        onPickSuggestion={vi.fn<(text: string) => void>()}
        status="streaming"
      />,
    );

    expect(
      screen.queryByRole('navigation', { name: 'Следни прашања' }),
    ).not.toBeInTheDocument();
  });
});

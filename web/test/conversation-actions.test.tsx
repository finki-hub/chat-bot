import type { ReactElement } from 'react';

import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { AnswerActionsProps } from '@/components/chat/answer-actions';
import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import {
  type ChatStatus,
  renderAnswerActions,
} from '@/lib/conversation-actions';

const REGENERATE = 'Регенерирај';

const assistant = (responseId: string): MyUIMessage => ({
  id: 'a1',
  metadata: { feedback: 'like', responseId },
  parts: [{ text: 'Одговор', type: 'text' }],
  role: 'assistant',
});

const renderFor = (
  responseId: string,
  disabled = false,
  status: ChatStatus = 'ready',
): ReactElement<AnswerActionsProps> => {
  const message = assistant(responseId);

  return renderAnswerActions({
    disabled,
    messages: [message],
    onVote: vi.fn<(messageId: string, vote: FeedbackType) => void>(),
    regenerate: vi.fn<(options: { messageId: string }) => void>(),
    status,
  })(message) as ReactElement<AnswerActionsProps>;
};

describe('renderAnswerActions', () => {
  it('keys the actions by responseId so a regenerated answer remounts and resets the vote', () => {
    expect(renderFor('r-old').key).not.toBe(renderFor('r-new').key);
    expect(renderFor('r-new').key).toContain('r-new');
  });

  it('drops the regenerate action when the backend is unavailable', () => {
    expect(renderFor('r1', false).props.onRegenerate).toBeDefined();
    expect(renderFor('r1', true).props.onRegenerate).toBeUndefined();
  });

  it('keeps the regenerate button mounted but disabled while streaming', () => {
    render(renderFor('r1', false, 'streaming'));

    expect(screen.getByRole('button', { name: REGENERATE })).toBeDisabled();
  });

  it('enables the regenerate button when idle', () => {
    render(renderFor('r1', false, 'ready'));

    expect(screen.getByRole('button', { name: REGENERATE })).toBeEnabled();
  });

  it('drops the action bar for a text-less (errored) turn', () => {
    const message: MyUIMessage = {
      id: 'a1',
      metadata: {
        error: { code: 'agent_error', message: 'грешка' },
        responseId: 'r1',
      },
      parts: [],
      role: 'assistant',
    };
    const node = renderAnswerActions({
      disabled: false,
      messages: [message],
      onVote: vi.fn<(messageId: string, vote: FeedbackType) => void>(),
      regenerate: vi.fn<(options: { messageId: string }) => void>(),
      status: 'ready',
    })(message);

    expect(node).toBeNull();
  });
});

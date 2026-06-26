import type { ReactElement } from 'react';

import { describe, expect, it, vi } from 'vitest';

import type { AnswerActionsProps } from '@/components/chat/answer-actions';
import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { renderAnswerActions } from '@/lib/conversation-actions';

const assistant = (responseId: string): MyUIMessage => ({
  id: 'a1',
  metadata: { feedback: 'like', responseId },
  parts: [{ text: 'Одговор', type: 'text' }],
  role: 'assistant',
});

const renderFor = (
  responseId: string,
  disabled = false,
): ReactElement<AnswerActionsProps> => {
  const message = assistant(responseId);

  return renderAnswerActions({
    disabled,
    messages: [message],
    onVote: vi.fn<(messageId: string, vote: FeedbackType) => void>(),
    regenerate: vi.fn<(options: { messageId: string }) => void>(),
    status: 'ready',
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
});

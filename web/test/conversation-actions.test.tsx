import type { ReactElement } from 'react';

import { describe, expect, it, vi } from 'vitest';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { renderAnswerActions } from '@/lib/conversation-actions';

const assistant = (responseId: string): MyUIMessage => ({
  id: 'a1',
  metadata: { feedback: 'like', responseId },
  parts: [{ text: 'Одговор', type: 'text' }],
  role: 'assistant',
});

const keyFor = (responseId: string): null | string => {
  const message = assistant(responseId);
  const element = renderAnswerActions({
    messages: [message],
    onVote: vi.fn<(messageId: string, vote: FeedbackType) => void>(),
    regenerate: vi.fn<(options: { messageId: string }) => void>(),
    status: 'ready',
  })(message) as ReactElement;

  return element.key;
};

describe('renderAnswerActions', () => {
  it('keys the actions by responseId so a regenerated answer remounts and resets the vote', () => {
    expect(keyFor('r-old')).not.toBe(keyFor('r-new'));
    expect(keyFor('r-new')).toContain('r-new');
  });
});

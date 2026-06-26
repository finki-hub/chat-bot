'use client';

import type { ReactNode } from 'react';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';
import { joinText } from '@/lib/message-parts';

export type ChatStatus = 'error' | 'ready' | 'streaming' | 'submitted';

const priorUserText = (
  messages: MyUIMessage[],
  message: MyUIMessage,
): string | undefined => {
  const prior = messages
    .slice(0, messages.indexOf(message))
    .findLast((m) => m.role === 'user');

  return prior ? joinText(prior) : undefined;
};

export type AnswerActionsContext = {
  messages: MyUIMessage[];
  onVote: (messageId: string, vote: FeedbackType) => void;
  regenerate: (options: { messageId: string }) => void;
  status: ChatStatus;
};

export const renderAnswerActions =
  ({ messages, onVote, regenerate, status }: AnswerActionsContext) =>
  (message: MyUIMessage): ReactNode => {
    if (message.role !== 'assistant') {
      return null;
    }

    const streaming = status === 'streaming' || status === 'submitted';

    return (
      <AnswerActions
        message={message}
        onRegenerate={
          streaming
            ? undefined
            : () => {
                regenerate({ messageId: message.id });
              }
        }
        onVote={(vote) => {
          onVote(message.id, vote);
        }}
        pending={streaming && messages.at(-1)?.id === message.id}
        questionText={priorUserText(messages, message)}
      />
    );
  };

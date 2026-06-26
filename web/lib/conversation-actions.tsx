'use client';

import type { ReactNode } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

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

export const renderAnswerActions =
  (
    messages: MyUIMessage[],
    regenerate: (options: { messageId: string }) => void,
    status: ChatStatus,
  ) =>
  (message: MyUIMessage): ReactNode =>
    message.role === 'assistant' ? (
      <AnswerActions
        message={message}
        onRegenerate={
          status === 'streaming' || status === 'submitted'
            ? undefined
            : () => {
                regenerate({ messageId: message.id });
              }
        }
        questionText={priorUserText(messages, message)}
      />
    ) : null;

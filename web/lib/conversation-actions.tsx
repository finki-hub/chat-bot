'use client';

import type { ReactNode } from 'react';

import { posthog } from 'posthog-js';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';
import { hasText, joinText } from '@/lib/message-parts';

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
  disabled: boolean;
  messages: MyUIMessage[];
  onVote: (messageId: string, vote: FeedbackType) => void;
  regenerate: (options: { messageId: string }) => void;
  status: ChatStatus;
};

export const renderAnswerActions =
  ({ disabled, messages, onVote, regenerate, status }: AnswerActionsContext) =>
  (message: MyUIMessage): ReactNode => {
    // No answer text means an errored/empty turn — skip the bar so it can't be voted on.
    if (message.role !== 'assistant' || !hasText(message)) {
      return null;
    }

    const streaming = status === 'streaming' || status === 'submitted';

    return (
      <AnswerActions
        key={`${message.id}:${message.metadata?.responseId ?? ''}`}
        message={message}
        // Keep the button mounted while streaming (disabled below) and only drop it
        // when the backend is unavailable, so regeneration isn't even offered then.
        onRegenerate={
          disabled
            ? undefined
            : () => {
                /* eslint-disable camelcase -- PostHog event properties are snake_case. */
                posthog.capture('chat_regenerated', {
                  inference_model: message.metadata?.inferenceModel,
                  message_index: messages.indexOf(message),
                  response_id: message.metadata?.responseId,
                });
                /* eslint-enable camelcase -- end of PostHog snake_case properties. */
                regenerate({ messageId: message.id });
              }
        }
        onVote={(vote) => {
          onVote(message.id, vote);
        }}
        pending={streaming && messages.at(-1)?.id === message.id}
        questionText={priorUserText(messages, message)}
        regenerateDisabled={streaming}
      />
    );
  };

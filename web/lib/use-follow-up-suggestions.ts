'use client';

import { useEffect, useRef, useState } from 'react';

import type { ErrorNotice, ModelId, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { fetchChatSuggestions } from '@/lib/chat-suggestions';
import { answerText, joinText } from '@/lib/message-parts';

export type FollowUps = {
  readonly messageId: string;
  readonly suggestions: readonly string[];
};

type UseFollowUpSuggestionsOptions = {
  readonly activeError: ErrorNotice | undefined;
  readonly conversationId: null | string;
  readonly messages: readonly MyUIMessage[];
  readonly model: ModelId;
  readonly status: 'error' | 'ready' | 'streaming' | 'submitted';
};

// Follow-up questions for an answer that just finished streaming here; opening
// an older conversation asks for none.
export const useFollowUpSuggestions = ({
  activeError,
  conversationId,
  messages,
  model,
  status,
}: UseFollowUpSuggestionsOptions): FollowUps | null => {
  const [followUps, setFollowUps] = useState<FollowUps | null>(null);
  const previousStatusRef = useRef(status);
  const conversationRef = useRef(conversationId);
  conversationRef.current = conversationId;

  useEffect(() => {
    setFollowUps(null);
  }, [conversationId]);

  useEffect(() => {
    const previous = previousStatusRef.current;
    previousStatusRef.current = status;
    if (status === 'submitted' || status === 'streaming') {
      setFollowUps(null);
      return;
    }
    const justFinished = previous === 'streaming' || previous === 'submitted';
    if (
      status !== 'ready' ||
      activeError !== undefined ||
      conversationId === null ||
      !justFinished
    ) {
      return;
    }

    const answer = messages.at(-1);
    const question = messages.at(-2);
    if (
      answer?.role !== 'assistant' ||
      question?.role !== 'user' ||
      answer.metadata?.error !== undefined
    ) {
      return;
    }
    const answerContent = answerText(answer)?.trim();
    const questionContent = joinText(question).trim();
    if (!answerContent || !questionContent) {
      return;
    }

    const run = async (): Promise<void> => {
      const suggestions = await fetchChatSuggestions({
        answer: answerContent,
        conversationId,
        providerModel: model,
        question: questionContent,
      });
      if (
        suggestions.length > 0 &&
        conversationRef.current === conversationId
      ) {
        setFollowUps({ messageId: answer.id, suggestions });
      }
    };
    fireAndForget(run());
  }, [activeError, conversationId, messages, model, status]);

  return followUps;
};

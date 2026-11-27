'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

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
// an older conversation or stopping the answer asks for none.
export const useFollowUpSuggestions = ({
  activeError,
  conversationId,
  messages,
  model,
  status,
}: UseFollowUpSuggestionsOptions): {
  readonly followUps: FollowUps | null;
  readonly markStopped: () => void;
} => {
  const [followUps, setFollowUps] = useState<FollowUps | null>(null);
  const previousStatusRef = useRef(status);
  const stoppedRef = useRef(false);
  // Bumped by every new turn or conversation, so a late reply is dropped.
  const turnRef = useRef(0);

  useEffect(() => {
    turnRef.current += 1;
    setFollowUps(null);
  }, [conversationId]);

  const markStopped = useCallback(() => {
    stoppedRef.current = true;
  }, []);

  useEffect(() => {
    const previous = previousStatusRef.current;
    previousStatusRef.current = status;
    if (status === 'submitted' || status === 'streaming') {
      if (previous !== 'submitted' && previous !== 'streaming') {
        turnRef.current += 1;
        stoppedRef.current = false;
      }
      setFollowUps(null);
      return;
    }
    const justFinished = previous === 'streaming' || previous === 'submitted';
    if (
      !justFinished ||
      status !== 'ready' ||
      activeError !== undefined ||
      conversationId === null ||
      stoppedRef.current
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

    const turn = turnRef.current;
    const run = async (): Promise<void> => {
      const suggestions = await fetchChatSuggestions({
        answer: answerContent,
        conversationId,
        providerModel: model,
        question: questionContent,
      });
      if (suggestions.length > 0 && turnRef.current === turn) {
        setFollowUps({ messageId: answer.id, suggestions });
      }
    };
    fireAndForget(run());
  }, [activeError, conversationId, messages, model, status]);

  return { followUps, markStopped };
};

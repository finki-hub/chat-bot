'use client';

import { type RefObject, useEffect } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { hasText } from '@/lib/message-parts';

const hasAssistantText = (message: MyUIMessage): boolean =>
  message.role === 'assistant' && hasText(message);

export const useStreamTiming = ({
  firstTokenAtRef,
  messages,
  setStreamStartedAt,
  startedAtRef,
  status,
}: {
  firstTokenAtRef: RefObject<null | number>;
  messages: MyUIMessage[];
  setStreamStartedAt: (value: null | number) => void;
  startedAtRef: RefObject<null | number>;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
}): void => {
  useEffect(() => {
    if (status === 'submitted') {
      const now = Date.now();
      startedAtRef.current = now;
      firstTokenAtRef.current = null;
      setStreamStartedAt(now);
    } else if (status === 'ready' || status === 'error') {
      setStreamStartedAt(null);
    }
  }, [status, firstTokenAtRef, setStreamStartedAt, startedAtRef]);

  useEffect(() => {
    if (firstTokenAtRef.current !== null || startedAtRef.current === null) {
      return;
    }

    const last = messages.at(-1);

    if (last !== undefined && hasAssistantText(last)) {
      firstTokenAtRef.current = Date.now();
    }
  }, [messages, firstTokenAtRef, startedAtRef]);
};

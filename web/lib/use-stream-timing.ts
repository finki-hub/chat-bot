'use client';

import { type RefObject, useEffect } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { hasText } from '@/lib/message-parts';

const hasAssistantText = (message: MyUIMessage): boolean =>
  message.role === 'assistant' && hasText(message);

export const useStreamTiming = ({
  firstTokenAtRef,
  messages,
  startedAtRef,
  status,
}: {
  firstTokenAtRef: RefObject<null | number>;
  messages: MyUIMessage[];
  startedAtRef: RefObject<null | number>;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
}): void => {
  useEffect(() => {
    if (status !== 'submitted') {
      return;
    }
    startedAtRef.current = Date.now();
    firstTokenAtRef.current = null;
  }, [status, firstTokenAtRef, startedAtRef]);

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

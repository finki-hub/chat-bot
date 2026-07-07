'use client';

import { posthog } from 'posthog-js';
import { type RefObject, useCallback } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { stopOptionsFrom } from '@/lib/stop-chat-snapshot';
import { stopChatStream } from '@/lib/transport';

type UseStopChatOptions = {
  readonly convoIdRef: RefObject<null | string>;
  readonly messages: readonly MyUIMessage[];
  readonly model: string;
  readonly stop: () => Promise<void> | void;
};

export const useStopChat = ({
  convoIdRef,
  messages,
  model,
  stop,
}: UseStopChatOptions) =>
  useCallback(() => {
    /* eslint-disable camelcase -- PostHog event properties are snake_case. */
    posthog.capture('chat_stopped', {
      inference_model: model,
    });
    /* eslint-enable camelcase -- end of PostHog snake_case properties. */
    const stopCurrent = async (): Promise<void> => {
      const cid = convoIdRef.current;
      if (cid === null) {
        await stop();
        return;
      }
      try {
        await stopChatStream(cid, stopOptionsFrom(messages));
      } finally {
        await stop();
      }
    };

    fireAndForget(stopCurrent());
  }, [convoIdRef, messages, model, stop]);

'use client';

import { posthog } from 'posthog-js';
import { type RefObject, useCallback } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { stopOptionsFrom } from '@/lib/stop-chat-snapshot';
import { stopChatStream } from '@/lib/transport';

type StopOrder = 'local-first' | 'server-first';

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
  useCallback(
    (order: StopOrder = 'server-first'): Promise<void> => {
      /* eslint-disable camelcase -- PostHog event properties are snake_case. */
      posthog.capture('chat_stopped', {
        inference_model: model,
      });
      /* eslint-enable camelcase -- end of PostHog snake_case properties. */
      const cid = convoIdRef.current;
      const snapshot = stopOptionsFrom(messages);
      const stopServer = async (): Promise<void> => {
        if (cid === null) {
          return;
        }

        await stopChatStream(cid, snapshot);
      };

      const stopCurrent = async (): Promise<void> => {
        if (order === 'local-first') {
          await stop();
          fireAndForget(stopServer());
          return;
        }

        try {
          await stopServer();
        } finally {
          await stop();
        }
      };

      return stopCurrent();
    },
    [convoIdRef, messages, model, stop],
  );

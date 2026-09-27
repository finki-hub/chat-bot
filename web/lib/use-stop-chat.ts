'use client';

import { posthog } from 'posthog-js';
import { type RefObject, useCallback } from 'react';

import type { ActiveChatRequest } from '@/lib/active-chat-request';

import { fireAndForget } from '@/lib/async';
import { stopChatStream } from '@/lib/transport';

export type StopOrder = 'local-first' | 'server-first';

type UseStopChatOptions = {
  readonly activeRequest: ActiveChatRequest;
  readonly convoIdRef: RefObject<null | string>;
  readonly model: string;
  readonly stop: () => Promise<void> | void;
};

export const useStopChat = ({
  activeRequest,
  convoIdRef,
  model,
  stop,
}: UseStopChatOptions) =>
  useCallback(
    (order: StopOrder = 'server-first'): Promise<void> => {
      const cid = convoIdRef.current;
      const current = activeRequest.takeSnapshot();
      const snapshot =
        activeRequest.conversationId === cid ? current : undefined;
      const responseId = snapshot?.responseId;
      /* eslint-disable camelcase -- PostHog event properties are snake_case. */
      posthog.capture('chat_stopped', {
        inference_model: model,
        ...(responseId !== undefined && { response_id: responseId }),
      });
      /* eslint-enable camelcase -- end of PostHog snake_case properties. */
      const stopServer = async (): Promise<void> => {
        if (cid === null || snapshot === undefined) {
          return;
        }

        // Preserve the existing server fallback while this active request's ID
        // is unknown. It targets the server's current stream, not a historical ID.
        await stopChatStream(
          cid,
          responseId === undefined ? undefined : { activeStreamId: responseId },
        );
      };

      const stopCurrent = async (): Promise<void> => {
        const stopResult = Promise.resolve(stop());

        if (order === 'local-first') {
          await stopResult;
          fireAndForget(stopServer());
          return;
        }

        try {
          await stopServer();
        } finally {
          await stopResult;
        }
      };

      return stopCurrent();
    },
    [activeRequest, convoIdRef, model, stop],
  );

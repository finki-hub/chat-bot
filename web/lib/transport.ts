import { DefaultChatTransport } from 'ai';
import { posthog } from 'posthog-js';

import type { ModelId, MyUIMessage, QueryTransformMode } from '@/lib/api-types';

export type ChatExtras = {
  embeddingsModel?: ModelId;
  maxTokens?: number;
  model: ModelId;
  queryTransformMode?: QueryTransformMode;
  queryTransformModel?: ModelId;
  reasoning?: boolean;
  temperature?: number;
  topP?: number;
};

export type StopChatStreamOptions = {
  readonly activeStreamId?: string;
};

export const buildChatTransport = (
  getExtras: () => ChatExtras,
): DefaultChatTransport<MyUIMessage> =>
  new DefaultChatTransport<MyUIMessage>({
    api: '/api/chat',
    prepareReconnectToStreamRequest: ({ id }) => ({
      api: `/api/chat/${encodeURIComponent(id)}/stream`,
    }),
    prepareSendMessagesRequest: ({ id, messageId, messages, trigger }) => ({
      body: {
        id,
        messageId,
        messages,
        trigger,
        ...getExtras(),
        posthogDistinctId: posthog.get_distinct_id(),
        posthogSessionId: posthog.get_session_id(),
      },
    }),
  });

export class StopChatStreamError extends Error {
  readonly status: number;

  constructor(status: number, options?: ErrorOptions) {
    super('Stop chat stream failed', options);
    this.name = 'StopChatStreamError';
    this.status = status;
  }
}

export const stopChatStream = async (
  conversationId: string,
  options?: StopChatStreamOptions,
): Promise<void> => {
  const init: RequestInit =
    options === undefined
      ? { method: 'POST' }
      : {
          body: JSON.stringify(options),
          headers: { 'content-type': 'application/json' },
          method: 'POST',
        };
  const response = await fetch(
    `/api/chat/${encodeURIComponent(conversationId)}/stop`,
    init,
  );

  if (!response.ok) {
    throw new StopChatStreamError(response.status);
  }
};

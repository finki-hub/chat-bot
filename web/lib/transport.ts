import { DefaultChatTransport } from 'ai';

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

export const stopChatStream = async (conversationId: string): Promise<void> => {
  const response = await fetch(
    `/api/chat/${encodeURIComponent(conversationId)}/stop`,
    {
      method: 'POST',
    },
  );

  if (!response.ok) {
    throw new StopChatStreamError(response.status);
  }
};

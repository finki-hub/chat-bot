import { DefaultChatTransport } from 'ai';

import type { ModelId, MyUIMessage, QueryTransformMode } from '@/lib/api-types';

import { getAnonUserId } from '@/lib/user';

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
      headers: { 'X-Client-User-Id': getAnonUserId() },
    }),
    prepareSendMessagesRequest: ({ id, messageId, messages, trigger }) => ({
      body: {
        id,
        messageId,
        messages,
        trigger,
        ...getExtras(),
        userId: getAnonUserId(),
      },
    }),
  });

export const stopChatStream = async (conversationId: string): Promise<void> => {
  await fetch(`/api/chat/${encodeURIComponent(conversationId)}/stop`, {
    headers: { 'X-Client-User-Id': getAnonUserId() },
    method: 'POST',
  });
};

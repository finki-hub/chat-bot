import { DefaultChatTransport } from 'ai';

// Extras are read lazily so the active model can change between turns without
// rebuilding the transport.
import type { ModelId, MyUIMessage } from '@/lib/api-types';

import { getAnonUserId } from '@/lib/user';

export type ChatExtras = {
  embeddingsModel?: ModelId;
  maxTokens?: number;
  model: ModelId;
  queryTransformModel?: ModelId;
  temperature?: number;
  topP?: number;
};

export const buildChatTransport = (
  getExtras: () => ChatExtras,
): DefaultChatTransport<MyUIMessage> =>
  new DefaultChatTransport<MyUIMessage>({
    api: '/api/chat',
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

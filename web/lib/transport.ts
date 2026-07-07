import { DefaultChatTransport } from 'ai';
import { posthog } from 'posthog-js';

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
    prepareSendMessagesRequest: ({ id, messageId, messages, trigger }) => ({
      body: {
        id,
        messageId,
        messages,
        trigger,
        ...getExtras(),
        posthogSessionId: posthog.get_session_id(),
        userId: getAnonUserId(),
      },
    }),
  });

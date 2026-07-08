import type { MyUIMessage } from '@/lib/api-types';
import type { StopChatStreamOptions } from '@/lib/transport';

export const stopOptionsFrom = (
  messages: readonly MyUIMessage[],
): StopChatStreamOptions | undefined => {
  const assistant = messages.findLast(
    (message) => message.role === 'assistant',
  );

  if (assistant === undefined) {
    return undefined;
  }

  const responseId = assistant.metadata?.responseId;

  return responseId === undefined ? undefined : { activeStreamId: responseId };
};

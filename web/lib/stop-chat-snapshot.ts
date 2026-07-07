import type { MyUIMessage } from '@/lib/api-types';
import type { StopChatStreamOptions } from '@/lib/transport';

import { joinText } from '@/lib/message-parts';

export const stopOptionsFrom = (
  messages: readonly MyUIMessage[],
): StopChatStreamOptions | undefined => {
  const assistant = messages.findLast(
    (message) => message.role === 'assistant',
  );

  if (assistant === undefined) {
    return undefined;
  }

  const content = joinText(assistant).trim();

  if (content.length === 0) {
    return undefined;
  }

  const responseId = assistant.metadata?.responseId;

  return {
    ...(responseId !== undefined && { activeStreamId: responseId }),
    assistantSnapshot: { content, metadata: assistant.metadata ?? {} },
  };
};

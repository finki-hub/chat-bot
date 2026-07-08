import type {
  ChatStateJsonValue,
  ChatStateMetadata,
} from '@/lib/chat-state-client';
import type { UiStreamMeta } from '@/lib/chat-translate';

import {
  type ConversationTurn,
  MAX_CHARS_PER_TURN,
  MAX_MESSAGES,
  type MyUIMessage,
} from '@/lib/api-types';
import { joinText } from '@/lib/message-parts';

type PersistedTurn = ConversationTurn & {
  readonly id: string;
};

type UserMessageForState = {
  readonly content: string;
  readonly id: string;
};

const isRecord = (
  value: ChatStateJsonValue,
): value is Record<string, ChatStateJsonValue> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isConversationRole = (
  value: unknown,
): value is ConversationTurn['role'] =>
  value === 'assistant' || value === 'user';

const persistedTurnFrom = (value: ChatStateJsonValue): null | PersistedTurn => {
  if (!isRecord(value)) {
    return null;
  }

  const content = value['content'];
  const id = value['id'];
  const role = value['role'];

  if (
    typeof content !== 'string' ||
    typeof id !== 'string' ||
    !isConversationRole(role)
  ) {
    return null;
  }

  return { content, id, role };
};

export const persistedTurns = (
  messages: readonly ChatStateJsonValue[],
  currentMessageId: string,
  requestTurnCount: number,
): ConversationTurn[] => {
  const turns = messages.flatMap((message) => {
    const turn = persistedTurnFrom(message);

    return turn === null ? [] : [turn];
  });
  const currentIndex = turns.findIndex((turn) => turn.id === currentMessageId);
  const previousTurns =
    currentIndex === -1 ? turns : turns.slice(0, currentIndex);
  const historyLimit = Math.max(0, MAX_MESSAGES - requestTurnCount);

  return previousTurns.slice(-historyLimit).map((turn) => ({
    content: turn.content.slice(0, MAX_CHARS_PER_TURN),
    role: turn.role,
  }));
};

export const lastUserMessageForState = (
  message: MyUIMessage | undefined,
): null | UserMessageForState => {
  if (message === undefined) {
    return null;
  }

  const content = joinText(message).trim();

  return content.length === 0 ? null : { content, id: message.id };
};

export const assistantMetadata = (meta: UiStreamMeta): ChatStateMetadata => ({
  ...(meta.inferenceModel !== undefined && {
    inferenceModel: meta.inferenceModel,
  }),
  ...(meta.responseId !== undefined && { responseId: meta.responseId }),
});

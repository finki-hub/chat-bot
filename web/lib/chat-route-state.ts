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

type AssistantState = {
  readonly metadata: ChatStateMetadata;
  readonly parts: readonly ChatStateJsonValue[];
};

type JsonValueResult =
  | { readonly success: false }
  | { readonly success: true; readonly value: ChatStateJsonValue };

type PersistedTurn = ConversationTurn & {
  readonly id: string;
};

type UserMessageForState = {
  readonly content: string;
  readonly id: string;
};

const isRecord = (
  value: unknown,
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
  const candidateTurns =
    currentIndex === -1 ? turns : turns.slice(0, currentIndex);
  const regenerationUserIndex =
    turns[currentIndex]?.role === 'assistant'
      ? candidateTurns.findLastIndex((turn) => turn.role === 'user')
      : null;

  if (regenerationUserIndex === -1) {
    throw new TypeError('Regeneration history requires a prior user turn');
  }

  const previousTurns =
    regenerationUserIndex === null
      ? candidateTurns
      : candidateTurns.slice(0, regenerationUserIndex + 1);
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

const jsonValueFrom = (value: unknown): JsonValueResult => {
  if (value === null) {
    return { success: true, value: null };
  }
  if (
    typeof value === 'boolean' ||
    typeof value === 'number' ||
    typeof value === 'string'
  ) {
    return { success: true, value };
  }
  if (Array.isArray(value)) {
    const parsedItems = value.flatMap((item) => {
      const parsed = jsonValueFrom(item);

      return parsed.success ? [parsed.value] : [];
    });
    return { success: true, value: parsedItems };
  }
  if (typeof value === 'object') {
    const parsedRecord = Object.fromEntries(
      Object.entries(value).flatMap(([key, item]) => {
        const parsed = jsonValueFrom(item);

        return parsed.success ? [[key, parsed.value]] : [];
      }),
    );
    return { success: true, value: parsedRecord };
  }

  return { success: false };
};

export const assistantState = (
  message: MyUIMessage,
  meta: UiStreamMeta,
): AssistantState => {
  const persistedMetadata = jsonValueFrom(message.metadata ?? {});
  const metadata =
    persistedMetadata.success && isRecord(persistedMetadata.value)
      ? persistedMetadata.value
      : {};
  const diagnostics = message.metadata?.diagnostics;
  const serverTotalMs = diagnostics?.serverTotalMs;
  const serverTtftMs = diagnostics?.serverTtftMs;
  const timing =
    message.metadata?.timing ??
    (typeof serverTotalMs === 'number'
      ? {
          totalMs: serverTotalMs,
          ttftMs: typeof serverTtftMs === 'number' ? serverTtftMs : null,
        }
      : undefined);

  return {
    metadata: {
      ...metadata,
      ...(meta.inferenceModel !== undefined && {
        inferenceModel: meta.inferenceModel,
      }),
      ...(meta.responseId !== undefined && { responseId: meta.responseId }),
      ...(timing !== undefined && { timing }),
    },
    parts: message.parts.flatMap((part) => {
      const parsed = jsonValueFrom(part);

      return parsed.success ? [parsed.value] : [];
    }),
  };
};

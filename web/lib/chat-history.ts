import { safeValidateUIMessages } from 'ai';

import type { MyUIMessage } from '@/lib/api-types';
import type { ChatStateJsonValue } from '@/lib/chat-state-client';

type ChatStateMessage = {
  readonly content: string;
  readonly id: string;
  readonly metadata: ChatStateJsonValue;
  readonly parts: null | readonly ChatStateJsonValue[];
  readonly responseId: null | string;
  readonly role: 'assistant' | 'user';
};

const isRecord = (
  value: ChatStateJsonValue | undefined,
): value is Readonly<Record<string, ChatStateJsonValue>> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const metadataFrom = (
  metadata: ChatStateJsonValue,
  responseId: null | string,
): Record<string, ChatStateJsonValue> => {
  const base = isRecord(metadata) ? metadata : {};
  const persistedResponseId = base['responseId'];
  const diagnostics = base['diagnostics'];
  const serverTotalMs = isRecord(diagnostics)
    ? diagnostics['serverTotalMs']
    : undefined;
  const serverTtftMs = isRecord(diagnostics)
    ? diagnostics['serverTtftMs']
    : undefined;
  const derivedTiming =
    base['timing'] === undefined && typeof serverTotalMs === 'number'
      ? {
          totalMs: serverTotalMs,
          ttftMs: typeof serverTtftMs === 'number' ? serverTtftMs : null,
        }
      : undefined;

  return {
    ...base,
    ...(responseId !== null && { responseId }),
    ...(responseId === null &&
      typeof persistedResponseId === 'string' && {
        responseId: persistedResponseId,
      }),
    ...(derivedTiming !== undefined && { timing: derivedTiming }),
  };
};

const messageFrom = async (
  message: ChatStateMessage,
): Promise<MyUIMessage | null> => {
  const fallbackParts = [{ text: message.content, type: 'text' }] as const;
  const parts =
    message.parts === null || message.parts.length === 0
      ? fallbackParts
      : message.parts;
  const validation = await safeValidateUIMessages<MyUIMessage>({
    messages: [
      {
        id: message.id,
        metadata: metadataFrom(message.metadata, message.responseId),
        parts,
        role: message.role,
      },
    ],
  });
  if (validation.success) {
    return validation.data[0] ?? null;
  }

  const fallbackValidation = await safeValidateUIMessages<MyUIMessage>({
    messages: [
      {
        id: message.id,
        metadata: metadataFrom(message.metadata, message.responseId),
        parts: fallbackParts,
        role: message.role,
      },
    ],
  });
  return fallbackValidation.success
    ? (fallbackValidation.data[0] ?? null)
    : null;
};

const parseMessage = (value: ChatStateJsonValue): ChatStateMessage | null => {
  if (!isRecord(value)) {
    return null;
  }
  const content = value['content'];
  const id = value['id'];
  const metadata = value['metadata'];
  const persistedParts = value['parts'];
  const parts = Array.isArray(persistedParts) ? persistedParts : null;
  const responseId = value['response_id'];
  const role = value['role'];
  if (
    typeof content !== 'string' ||
    typeof id !== 'string' ||
    metadata === undefined ||
    (responseId !== null && typeof responseId !== 'string') ||
    (role !== 'assistant' && role !== 'user')
  ) {
    return null;
  }
  return { content, id, metadata, parts, responseId, role };
};

export const parseChatStateMessages = async (
  messages: readonly ChatStateJsonValue[],
): Promise<MyUIMessage[]> => {
  const parsed = await Promise.all(
    messages.map(async (message) => {
      const stateMessage = parseMessage(message);
      return stateMessage === null ? null : messageFrom(stateMessage);
    }),
  );
  return parsed.flatMap((message) => (message === null ? [] : [message]));
};

export const sanitizeSharedMessages = (
  messages: readonly MyUIMessage[],
): MyUIMessage[] =>
  messages.map((message) => ({
    id: message.id,
    metadata:
      message.metadata?.sources === undefined
        ? {}
        : { sources: message.metadata.sources },
    parts: message.parts,
    role: message.role,
  }));

import { safeValidateUIMessages } from 'ai';

import type { ModelId, MyUIMessage } from '@/lib/api-types';
import type { ChatStateJsonValue } from '@/lib/chat-state-client';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';

/* eslint-disable camelcase -- Python chat state API uses snake_case fields. */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type ChatStateMessage = {
  readonly content: string;
  readonly id: string;
  readonly metadata: ChatStateJsonValue;
  readonly parts: null | readonly ChatStateJsonValue[];
  readonly response_id: null | string;
  readonly role: 'assistant' | 'user';
};

type HistoryConversation = {
  readonly id: string;
  readonly model: ModelId | null;
  readonly title: null | string;
};

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

const empty = (status: number): Response => new Response(null, { status });

const isRecord = (
  value: unknown,
): value is Record<string, ChatStateJsonValue> =>
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
        metadata: metadataFrom(message.metadata, message.response_id),
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
        metadata: metadataFrom(message.metadata, message.response_id),
        parts: fallbackParts,
        role: message.role,
      },
    ],
  });

  return fallbackValidation.success
    ? (fallbackValidation.data[0] ?? null)
    : null;
};

const isMessageRole = (value: unknown): value is ChatStateMessage['role'] =>
  value === 'assistant' || value === 'user';

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
    !isMessageRole(role)
  ) {
    return null;
  }

  return { content, id, metadata, parts, response_id: responseId, role };
};

export const GET = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  const { ChatStateRequestError, createChatStateClient } =
    await import('@/lib/chat-state-client');
  const chatState = createChatStateClient();

  try {
    const userId = await getAuthenticatedChatUserId();
    const { conversation, messages } = await chatState.loadConversation({
      conversationId,
      userId,
    });
    const parsedMessages = await Promise.all(
      messages.map(async (message) => {
        const parsed = parseMessage(message);

        return parsed === null ? null : messageFrom(parsed);
      }),
    );
    const uiMessages = parsedMessages.flatMap((message) =>
      message === null ? [] : [message],
    );
    const historyConversation: HistoryConversation = {
      id: conversation.id,
      model: conversation.model ?? null,
      title: conversation.title ?? null,
    };

    return Response.json({
      conversation: historyConversation,
      messages: uiMessages,
    });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return empty(401);
    }
    if (error instanceof ChatStateRequestError) {
      return empty(error.status);
    }

    throw error;
  }
};

/* eslint-enable camelcase -- end Python chat state API snake_case fields. */

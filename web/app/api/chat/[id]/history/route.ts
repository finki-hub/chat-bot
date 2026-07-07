import type { ModelId, MyMetadata, MyUIMessage } from '@/lib/api-types';
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
  value: ChatStateJsonValue,
): value is Record<string, ChatStateJsonValue> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const metadataFrom = (
  metadata: ChatStateJsonValue,
  responseId: null | string,
): MyMetadata => {
  const base = isRecord(metadata) ? metadata : {};
  const feedback = base['feedback'];
  const inferenceModel = base['inferenceModel'];
  const persistedResponseId = base['responseId'];
  const feedbackMetadata: Pick<MyMetadata, 'feedback'> =
    feedback === 'dislike' || feedback === 'like' ? { feedback } : {};

  return {
    ...feedbackMetadata,
    ...(typeof inferenceModel === 'string' && { inferenceModel }),
    ...(responseId !== null && { responseId }),
    ...(responseId === null &&
      typeof persistedResponseId === 'string' && {
        responseId: persistedResponseId,
      }),
  };
};

const messageFrom = (message: ChatStateMessage): MyUIMessage => ({
  id: message.id,
  metadata: metadataFrom(message.metadata, message.response_id),
  parts: [{ text: message.content, type: 'text' }],
  role: message.role,
});

const isMessageRole = (value: unknown): value is ChatStateMessage['role'] =>
  value === 'assistant' || value === 'user';

const parseMessage = (value: ChatStateJsonValue): ChatStateMessage | null => {
  if (!isRecord(value)) {
    return null;
  }

  const content = value['content'];
  const id = value['id'];
  const metadata = value['metadata'];
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

  return { content, id, metadata, response_id: responseId, role };
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
    const uiMessages = messages.flatMap((message) => {
      const parsed = parseMessage(message);

      return parsed === null ? [] : [messageFrom(parsed)];
    });
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

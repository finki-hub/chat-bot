import type { ModelId } from '@/lib/api-types';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import { parseChatStateMessages } from '@/lib/chat-history';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type HistoryConversation = {
  readonly id: string;
  readonly model: ModelId | null;
  readonly title: null | string;
};

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

const empty = (status: number): Response => new Response(null, { status });

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
    const uiMessages = await parseChatStateMessages(messages);
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

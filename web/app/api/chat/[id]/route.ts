import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

export const DELETE = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  const chatState = createChatStateClient();

  try {
    const userId = await getAuthenticatedChatUserId();
    await chatState.deleteConversation({ conversationId, userId });

    return new Response(null, { status: 204 });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return new Response(null, { status: 401 });
    }
    if (error instanceof ChatStateRequestError) {
      return new Response(null, { status: error.status });
    }

    throw error;
  }
};

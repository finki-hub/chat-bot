import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import { createChatSharingClient } from '@/lib/chat-sharing-client';
import { ChatStateRequestError } from '@/lib/chat-state-client';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

const empty = (status: number): Response => new Response(null, { status });

export const POST = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  try {
    const userId = await getAuthenticatedChatUserId();
    const share = await createChatSharingClient().createConversationShare({
      conversationId,
      userId,
    });
    return Response.json(share);
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

export const GET = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  try {
    const userId = await getAuthenticatedChatUserId();
    const shared = await createChatSharingClient().getConversationShareStatus({
      conversationId,
      userId,
    });
    return empty(shared ? 200 : 204);
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

export const DELETE = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  try {
    const userId = await getAuthenticatedChatUserId();
    await createChatSharingClient().revokeConversationShare({
      conversationId,
      userId,
    });
    return empty(204);
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

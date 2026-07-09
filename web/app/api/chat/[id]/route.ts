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

type UpdateConversationPayload = {
  readonly expectedTitle?: string;
  readonly model?: string;
  readonly title?: string;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const textValue = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim().length > 0
    ? value.trim()
    : undefined;

const parseUpdatePayload = async (
  request: Request,
): Promise<null | UpdateConversationPayload> => {
  let body: unknown;
  try {
    body = await request.json();
  } catch (error) {
    if (error instanceof SyntaxError) {
      return null;
    }
    throw error;
  }
  if (!isRecord(body)) {
    return null;
  }
  const expectedTitle = textValue(body['expectedTitle']);
  const model = textValue(body['model']);
  const title = textValue(body['title']);
  const payload: UpdateConversationPayload = {
    ...(expectedTitle !== undefined && { expectedTitle }),
    ...(model !== undefined && { model }),
    ...(title !== undefined && { title }),
  };
  return payload.model === undefined && payload.title === undefined
    ? null
    : payload;
};

const empty = (status: number): Response => new Response(null, { status });

export const PATCH = async (
  request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const payload = await parseUpdatePayload(request);
  if (payload === null) {
    return empty(400);
  }

  const { id: conversationId } = await params;
  const chatState = createChatStateClient();

  try {
    const userId = await getAuthenticatedChatUserId();
    if (payload.expectedTitle !== undefined) {
      const { conversation } = await chatState.loadConversation({
        conversationId,
        userId,
      });
      if (conversation.title !== payload.expectedTitle) {
        return empty(204);
      }
    }
    if (payload.expectedTitle === undefined && payload.model !== undefined) {
      await chatState.upsertConversation({
        conversationId,
        model: payload.model,
        ...(payload.title !== undefined && { title: payload.title }),
        userId,
      });
    } else {
      await chatState.updateConversation({
        conversationId,
        ...(payload.model !== undefined && { model: payload.model }),
        ...(payload.title !== undefined && { title: payload.title }),
        userId,
      });
    }

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

export const DELETE = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  const chatState = createChatStateClient();

  try {
    const userId = await getAuthenticatedChatUserId();
    await chatState.deleteConversation({ conversationId, userId });

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

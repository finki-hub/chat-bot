import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';
import { activeChatProducers } from '@/lib/resumable-stream-context';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

type StopBody = {
  readonly activeStreamId: null | string;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseStopBody = async (request: Request): Promise<StopBody> => {
  const text = await request.text();

  if (text.length === 0) {
    return { activeStreamId: null };
  }

  let payload: unknown;

  try {
    payload = JSON.parse(text) as unknown;
  } catch (error) {
    if (error instanceof SyntaxError) {
      return { activeStreamId: null };
    }
    throw error;
  }

  if (!isRecord(payload)) {
    return { activeStreamId: null };
  }

  const activeStreamId =
    typeof payload['activeStreamId'] === 'string' &&
    payload['activeStreamId'].length > 0
      ? payload['activeStreamId']
      : null;
  return { activeStreamId };
};

const ignoreMissing = async (operation: Promise<void>): Promise<void> => {
  try {
    await operation;
  } catch (error) {
    if (error instanceof ChatStateRequestError && error.status === 404) {
      return;
    }

    throw error;
  }
};

export const POST = async (
  request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const [{ id: conversationId }, body] = await Promise.all([
    params,
    parseStopBody(request),
  ]);
  const chatState = createChatStateClient();

  try {
    const userId = await getAuthenticatedChatUserId();
    const { conversation } = await chatState.loadConversation({
      conversationId,
      userId,
    });
    const currentStreamId = conversation.active_stream_id;

    if (currentStreamId === null) {
      return Response.json({ aborted: false, stopped: false });
    }

    const requestedStreamId = body.activeStreamId ?? currentStreamId;

    if (requestedStreamId !== currentStreamId) {
      return Response.json({ aborted: false, stopped: false });
    }

    const aborted = activeChatProducers.abort(requestedStreamId);

    await ignoreMissing(
      chatState.stopActiveStreamIfCurrent({
        conversationId,
        streamId: requestedStreamId,
        userId,
      }),
    );
    await ignoreMissing(
      chatState.clearActiveStreamIfCurrent({
        conversationId,
        streamId: requestedStreamId,
        userId,
      }),
    );

    return Response.json({ aborted, stopped: true });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return Response.json({ aborted: false, stopped: false }, { status: 401 });
    }
    if (error instanceof ChatStateRequestError) {
      return new Response(null, { status: error.status });
    }

    throw error;
  }
};

import { UI_MESSAGE_STREAM_HEADERS } from 'ai';
import { after } from 'next/server';

import {
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';
import { createChatResumableStreamContext } from '@/lib/resumable-stream-context';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

const clientUserId = (request: Request): null | string => {
  const userId = request.headers.get('X-Client-User-Id');

  return userId === null || userId.length === 0 ? null : userId;
};

const empty = (status: number): Response => new Response(null, { status });

export const GET = async (
  request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const userId = clientUserId(request);

  if (userId === null) {
    return empty(400);
  }

  const { id: conversationId } = await params;
  const chatState = createChatStateClient();

  try {
    const { conversation } = await chatState.loadConversation({
      conversationId,
      userId,
    });
    const activeStreamId = conversation.active_stream_id;

    if (activeStreamId === null) {
      return empty(204);
    }

    const stream = await createChatResumableStreamContext({
      waitUntil: after,
    }).resumeExistingStream(activeStreamId);

    if (stream === null || stream === undefined) {
      await chatState.clearActiveStreamIfCurrent({
        conversationId,
        streamId: activeStreamId,
        userId,
      });

      return empty(204);
    }

    return new Response(stream, { headers: UI_MESSAGE_STREAM_HEADERS });
  } catch (error) {
    if (error instanceof ChatStateRequestError) {
      return empty(error.status);
    }

    throw error;
  }
};

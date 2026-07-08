import { UI_MESSAGE_STREAM_HEADERS } from 'ai';
import { after } from 'next/server';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
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

const empty = (status: number): Response => new Response(null, { status });

export const GET = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { id: conversationId } = await params;
  const chatState = createChatStateClient();

  try {
    const userId = await getAuthenticatedChatUserId();
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
      try {
        await chatState.clearActiveStreamIfCurrent({
          conversationId,
          streamId: activeStreamId,
          userId,
        });
      } catch (error) {
        if (!(error instanceof ChatStateRequestError && error.status === 404)) {
          throw error;
        }
      }

      return empty(204);
    }

    return new Response(stream, { headers: UI_MESSAGE_STREAM_HEADERS });
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

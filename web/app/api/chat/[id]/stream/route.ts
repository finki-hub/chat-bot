import { UI_MESSAGE_STREAM_HEADERS } from 'ai';
import { after } from 'next/server';
import { setTimeout as delay } from 'node:timers/promises';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  ChatStateRequestError,
  createChatStateClient,
  isResumableChatStreamStatus,
} from '@/lib/chat-state-client';
import { createChatResumableStreamContext } from '@/lib/resumable-stream-context';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

const empty = (status: number): Response => new Response(null, { status });

const REGISTRATION_RETRY_DELAYS_MS = [50, 100, 250, 500, 1_000, 2_000] as const;

const resumeActiveStream = async (
  activeStatus: null | string,
  activeStreamId: string,
  streamContext: ReturnType<typeof createChatResumableStreamContext>,
) => {
  let stream = await streamContext.resumeExistingStream(activeStreamId);
  if (stream !== undefined || activeStatus !== 'pending') {
    return stream;
  }
  for (const delayMs of REGISTRATION_RETRY_DELAYS_MS) {
    await delay(delayMs);
    stream = await streamContext.resumeExistingStream(activeStreamId);
    if (stream !== undefined) {
      return stream;
    }
  }
  return stream;
};

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

    if (
      activeStreamId === null ||
      !isResumableChatStreamStatus(conversation.active_status)
    ) {
      return empty(204);
    }

    const streamContext = createChatResumableStreamContext({
      waitUntil: after,
    });
    const stream = await resumeActiveStream(
      conversation.active_status,
      activeStreamId,
      streamContext,
    );

    if (stream === undefined && conversation.active_status === 'pending') {
      return empty(204);
    }

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

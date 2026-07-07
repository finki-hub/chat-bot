import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  type ChatStateMetadata,
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';
import { activeChatProducers } from '@/lib/resumable-stream-context';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type AssistantSnapshot = {
  readonly content: string;
  readonly metadata: ChatStateMetadata;
};

type RouteContext = {
  readonly params: Promise<{ readonly id: string }>;
};

type StopBody = {
  readonly activeStreamId: null | string;
  readonly assistantSnapshot: AssistantSnapshot | null;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isJsonValue = (value: unknown): value is ChatStateMetadata[string] => {
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'number' ||
    typeof value === 'string'
  ) {
    return true;
  }

  if (Array.isArray(value)) {
    return value.every(isJsonValue);
  }

  return isRecord(value) && Object.values(value).every(isJsonValue);
};

const metadataFrom = (value: unknown): ChatStateMetadata => {
  if (!isRecord(value)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, ChatStateMetadata[string]] =>
        isJsonValue(entry[1]),
    ),
  );
};

const parseStopBody = async (request: Request): Promise<StopBody> => {
  const text = await request.text();

  if (text.length === 0) {
    return { activeStreamId: null, assistantSnapshot: null };
  }

  let payload: unknown;

  try {
    payload = JSON.parse(text) as unknown;
  } catch (error) {
    if (error instanceof SyntaxError) {
      return { activeStreamId: null, assistantSnapshot: null };
    }
    throw error;
  }

  if (!isRecord(payload)) {
    return { activeStreamId: null, assistantSnapshot: null };
  }

  const activeStreamId =
    typeof payload['activeStreamId'] === 'string' &&
    payload['activeStreamId'].length > 0
      ? payload['activeStreamId']
      : null;
  const snapshotPayload = payload['assistantSnapshot'];
  const assistantSnapshot =
    isRecord(snapshotPayload) &&
    typeof snapshotPayload['content'] === 'string' &&
    snapshotPayload['content'].trim().length > 0
      ? {
          content: snapshotPayload['content'].trim(),
          metadata: metadataFrom(snapshotPayload['metadata']),
        }
      : null;

  return { activeStreamId, assistantSnapshot };
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

    if (body.activeStreamId === null) {
      return Response.json({ aborted: false, stopped: false });
    }

    if (body.activeStreamId !== currentStreamId) {
      return Response.json({ aborted: false, stopped: false });
    }

    const responseId = conversation.active_response_id ?? currentStreamId;

    if (body.assistantSnapshot !== null) {
      await chatState.upsertAssistantMessage({
        content: body.assistantSnapshot.content,
        conversationId,
        metadata: {
          ...body.assistantSnapshot.metadata,
          responseId,
          stopped: true,
        },
        responseId,
        userId,
      });
    }

    const aborted = activeChatProducers.abort(currentStreamId);

    await ignoreMissing(
      chatState.stopActiveStreamIfCurrent({
        conversationId,
        streamId: currentStreamId,
        userId,
      }),
    );
    await ignoreMissing(
      chatState.clearActiveStreamIfCurrent({
        conversationId,
        streamId: currentStreamId,
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

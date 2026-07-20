import 'server-only';

import {
  type ChatStateClient,
  ChatStateRequestError,
} from '@/lib/chat-state-client';
import {
  activeChatProducers,
  createChatResumableStreamContext,
  type WaitUntil,
} from '@/lib/resumable-stream-context';

type ActiveStreamIdentity = {
  readonly chatState: ChatStateClient;
  readonly conversationId: string;
  readonly streamId: string;
  readonly userId: string;
};

type PendingStreamRegistration = ActiveStreamIdentity & {
  readonly replacementMessageId: null | string;
  readonly upstreamController: AbortController;
};

type ResumableStreamStart = ActiveStreamIdentity & {
  readonly sseStream: ReadableStream<string>;
  readonly upstreamController: AbortController;
  readonly waitUntil: WaitUntil;
};

export const clearChatActiveStream = async ({
  chatState,
  conversationId,
  streamId,
  userId,
}: ActiveStreamIdentity): Promise<void> => {
  try {
    await chatState.clearActiveStreamIfCurrent({
      conversationId,
      streamId,
      userId,
    });
  } catch (error) {
    if (error instanceof ChatStateRequestError && error.status === 404) {
      return;
    }
    throw error;
  }
};

export const finishChatStream = async (
  identity: ActiveStreamIdentity,
): Promise<void> => {
  try {
    await clearChatActiveStream(identity);
  } finally {
    activeChatProducers.unregister(identity.streamId);
  }
};

export const registerPendingChatStream = async ({
  chatState,
  conversationId,
  replacementMessageId,
  streamId,
  upstreamController,
  userId,
}: PendingStreamRegistration): Promise<void> => {
  activeChatProducers.register(streamId, upstreamController);
  try {
    await chatState.setActiveStream({
      activeResponseId: streamId,
      activeStatus: 'pending',
      activeStreamId: streamId,
      conversationId,
      replacementMessageId,
      userId,
    });
  } catch (error) {
    upstreamController.abort();
    activeChatProducers.unregister(streamId);
    throw error;
  }
};

export const startResumableChatStream = async ({
  chatState,
  conversationId,
  sseStream,
  streamId,
  upstreamController,
  userId,
  waitUntil,
}: ResumableStreamStart): Promise<void> => {
  try {
    await createChatResumableStreamContext({
      waitUntil,
    }).createNewResumableStream(streamId, () => sseStream);
    try {
      await chatState.markActiveStreamStreamingIfPending({
        conversationId,
        streamId,
        userId,
      });
    } catch (error) {
      if (error instanceof ChatStateRequestError && error.status === 404) {
        return;
      }
      throw error;
    }
  } catch {
    upstreamController.abort();
    await finishChatStream({ chatState, conversationId, streamId, userId });
  }
};

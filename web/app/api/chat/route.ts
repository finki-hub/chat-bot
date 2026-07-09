import { createUIMessageStream, createUIMessageStreamResponse } from 'ai';
import { after } from 'next/server';

import type { MyUIMessage } from '@/lib/api-types';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  assistantMetadata,
  lastUserMessageForState,
  persistedTurns,
} from '@/lib/chat-route-state';
import {
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';
import {
  type ChatClientBody,
  currentUserMessageForRequest,
  toChatRequestBody,
  translateToUiStream,
  type UiStreamMeta,
} from '@/lib/chat-translate';
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';
import { joinText } from '@/lib/message-parts';
import {
  activeChatProducers,
  createChatResumableStreamContext,
  normalizePythonResponseStreamId,
} from '@/lib/resumable-stream-context';
import { parseProtocolV2 } from '@/lib/sse';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const SSE_CONTENT_TYPE = 'text/event-stream';

type ResumableChatClientBody = ChatClientBody & {
  readonly id?: string;
};

const readDetail = async (response: Response): Promise<string> => {
  try {
    const body = (await response.json()) as { detail?: string };

    return body.detail ?? 'Request failed';
  } catch {
    return 'Request failed';
  }
};

const errorResponse = (meta: UiStreamMeta, code: string, message: string) => {
  const stream = createUIMessageStream<MyUIMessage>({
    execute: ({ writer }) => {
      writer.write({ messageMetadata: meta, type: 'start' });
      writer.write({
        data: { code, message },
        transient: true,
        type: 'data-error',
      });
    },
  });

  return createUIMessageStreamResponse({ stream });
};

const unauthenticated = (): Response =>
  Response.json({ error: 'Authentication required' }, { status: 401 });

const requiredText = (value: string | undefined): null | string =>
  value === undefined || value.length === 0 ? null : value;

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

export const POST = async (req: Request): Promise<Response> => {
  try {
    const clientBody = (await req.json()) as ResumableChatClientBody;
    const chatBody = toChatRequestBody(clientBody);
    const inferenceModel = chatBody.inference_model;
    const conversationId = requiredText(clientBody.id);
    const userId = await getAuthenticatedChatUserId();
    const userMessage = lastUserMessageForState(
      currentUserMessageForRequest(clientBody),
    );

    if (conversationId === null || userMessage === null) {
      return errorResponse(
        { inferenceModel },
        'malformed_input',
        'Missing conversation id or user message',
      );
    }

    const chatState = createChatStateClient();

    await chatState.upsertConversation({
      conversationId,
      ...(inferenceModel !== undefined && { model: inferenceModel }),
      userId,
    });
    const loadedConversation = await chatState.loadConversation({
      conversationId,
      userId,
    });
    const trustedHistory = persistedTurns(
      loadedConversation.messages,
      userMessage.id,
      chatBody.messages.length,
    );
    await chatState.upsertUserMessage({
      content: userMessage.content,
      conversationId,
      messageId: userMessage.id,
      userId,
    });
    const upstreamController = new AbortController();

    const upstream = await fetch(`${API_BASE_URL}/chat/`, {
      body: JSON.stringify({
        ...chatBody,
        messages: [...trustedHistory, ...chatBody.messages],
      }),
      headers: {
        'content-type': 'application/json',
        'x-api-key': CHAT_API_KEY,
        'X-Distinct-Id':
          typeof clientBody.posthogDistinctId === 'string' &&
          clientBody.posthogDistinctId.length > 0
            ? clientBody.posthogDistinctId
            : userId,
        ...(typeof clientBody.posthogSessionId === 'string' &&
          clientBody.posthogSessionId.length > 0 && {
            'X-PostHog-Session-Id': clientBody.posthogSessionId,
          }),
      },
      method: 'POST',
      signal: upstreamController.signal,
    });

    const contentType = upstream.headers.get('content-type') ?? '';

    if (!upstream.ok || !contentType.includes(SSE_CONTENT_TYPE)) {
      const message = await readDetail(upstream);

      return errorResponse({ inferenceModel }, 'pre_stream', message);
    }

    const responseId = upstream.headers.get('X-Response-Id') ?? undefined;
    const streamId = normalizePythonResponseStreamId(responseId);
    const upstreamBody = upstream.body;

    if (upstreamBody === null) {
      return errorResponse(
        { inferenceModel, responseId },
        'agent_error',
        'Empty stream from API',
      );
    }

    activeChatProducers.register(streamId, upstreamController);

    try {
      await chatState.setActiveStream({
        activeResponseId: streamId,
        activeStreamId: streamId,
        conversationId,
        userId,
      });
    } catch (error) {
      upstreamController.abort();
      activeChatProducers.unregister(streamId);
      throw error;
    }

    const meta = { inferenceModel, responseId: streamId };

    const stream = createUIMessageStream<MyUIMessage>({
      execute: async ({ writer }) => {
        await translateToUiStream(parseProtocolV2(upstreamBody), writer, meta);
      },
      onEnd: async ({ responseMessage }) => {
        try {
          const content = joinText(responseMessage).trim();

          if (content.length > 0) {
            await chatState.upsertAssistantMessage({
              content,
              conversationId,
              metadata: assistantMetadata(meta),
              responseId: streamId,
              userId,
            });
          }

          await ignoreMissing(
            chatState.clearActiveStreamIfCurrent({
              conversationId,
              streamId,
              userId,
            }),
          );
        } finally {
          activeChatProducers.unregister(streamId);
        }
      },
      // Generic: this string reaches the browser, so don't leak raw errors.
      onError: () => 'stream error',
    });

    return createUIMessageStreamResponse({
      consumeSseStream: async ({ stream: sseStream }) => {
        try {
          await createChatResumableStreamContext({
            waitUntil: after,
          }).createNewResumableStream(streamId, () => sseStream);
        } catch {
          upstreamController.abort();
          await chatState.clearActiveStreamIfCurrent({
            conversationId,
            streamId,
            userId,
          });
          activeChatProducers.unregister(streamId);
        }
      },
      stream,
    });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return unauthenticated();
    }

    return errorResponse({}, 'internal', 'Request failed');
  }
};

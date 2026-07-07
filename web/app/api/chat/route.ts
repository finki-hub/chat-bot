import { createUIMessageStream, createUIMessageStreamResponse } from 'ai';

import type { MyUIMessage } from '@/lib/api-types';

import {
  type ChatClientBody,
  toChatRequestBody,
  translateToUiStream,
  type UiStreamMeta,
} from '@/lib/chat-translate';
import { API_BASE_URL } from '@/lib/env';
import { parseProtocolV2 } from '@/lib/sse';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const SSE_CONTENT_TYPE = 'text/event-stream';

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

export const POST = async (req: Request): Promise<Response> => {
  try {
    const clientBody = (await req.json()) as ChatClientBody;
    const chatBody = toChatRequestBody(clientBody);
    const inferenceModel = chatBody.inference_model;

    const upstream = await fetch(`${API_BASE_URL}/chat/`, {
      body: JSON.stringify(chatBody),
      headers: {
        'content-type': 'application/json',
        // Forward the browser's anonymous id so server-side analytics share its distinct_id.
        ...(typeof clientBody.userId === 'string' &&
          clientBody.userId.length > 0 && {
            'X-Distinct-Id': clientBody.userId,
          }),
        ...(typeof clientBody.posthogSessionId === 'string' &&
          clientBody.posthogSessionId.length > 0 && {
            'X-PostHog-Session-Id': clientBody.posthogSessionId,
          }),
      },
      method: 'POST',
      // Propagate client aborts so stopping the chat tears down upstream generation.
      signal: req.signal,
    });

    const contentType = upstream.headers.get('content-type') ?? '';

    if (!upstream.ok || !contentType.includes(SSE_CONTENT_TYPE)) {
      const message = await readDetail(upstream);

      return errorResponse({ inferenceModel }, 'pre_stream', message);
    }

    const responseId = upstream.headers.get('X-Response-Id') ?? undefined;
    const upstreamBody = upstream.body;

    if (upstreamBody === null) {
      return errorResponse(
        { inferenceModel, responseId },
        'agent_error',
        'Empty stream from API',
      );
    }

    const stream = createUIMessageStream<MyUIMessage>({
      execute: async ({ writer }) => {
        await translateToUiStream(parseProtocolV2(upstreamBody), writer, {
          inferenceModel,
          responseId,
        });
      },
      // Generic: this string reaches the browser, so don't leak raw errors.
      onError: () => 'stream error',
    });

    return createUIMessageStreamResponse({ stream });
  } catch {
    return errorResponse({}, 'internal', 'Request failed');
  }
};

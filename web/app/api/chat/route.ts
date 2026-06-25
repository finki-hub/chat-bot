import { createUIMessageStream, createUIMessageStreamResponse } from 'ai';

// BFF: translate the client chat request -> Python ChatSchema, POST it to
// {API_BASE_URL}/chat/, then re-stream the protocol-v2 SSE answer as an AI SDK
// v5 UI-message-stream. Pre-stream JSON errors (422/503/500) are surfaced as a
// transient data-error rather than crashing the stream. Server-only: API_BASE_URL
// never reaches the browser; the route runs on Node (env + streaming fetch).
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
  const clientBody = (await req.json()) as ChatClientBody;
  const chatBody = toChatRequestBody(clientBody);
  const inferenceModel = chatBody.inference_model;

  const upstream = await fetch(`${API_BASE_URL}/chat/`, {
    body: JSON.stringify(chatBody),
    headers: { 'content-type': 'application/json' },
    method: 'POST',
  });

  const contentType = upstream.headers.get('content-type') ?? '';

  // Pre-stream JSON errors (422/503/500) are NOT SSE — branch before streaming.
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
    onError: (error) =>
      error instanceof Error ? error.message : 'stream error',
  });

  return createUIMessageStreamResponse({ stream });
};

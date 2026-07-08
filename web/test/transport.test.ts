import { describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import {
  buildChatTransport,
  type ChatExtras,
  stopChatStream,
  StopChatStreamError,
} from '@/lib/transport';

vi.mock('posthog-js', () => ({
  posthog: {
    // eslint-disable-next-line camelcase -- PostHog SDK method name.
    get_distinct_id: () => 'browser-distinct-id',
    // eslint-disable-next-line camelcase -- PostHog SDK method name.
    get_session_id: () => 'session-test-id',
  },
}));

const SUBMIT = 'submit-message' as const;
const ACTIVE_STREAM_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d22';
const GPT_MODEL = 'gpt-5.4-mini';

type PrepareArgs = {
  id: string;
  messageId?: string;
  messages: MyUIMessage[];
  trigger: 'regenerate-message' | 'submit-message';
};

type PrepareTransport = {
  prepareReconnectToStreamRequest: (args: { id: string }) => {
    api?: string;
    headers?: HeadersInit;
  };
  prepareSendMessagesRequest: (args: PrepareArgs) => {
    body: Record<string, unknown>;
  };
};

const getPrepare = (extras: ChatExtras) => {
  const transport = buildChatTransport(
    () => extras,
  ) as unknown as PrepareTransport;

  return transport.prepareSendMessagesRequest.bind(transport);
};

const sampleMessages: MyUIMessage[] = [
  { id: 'u1', parts: [{ text: 'здраво', type: 'text' }], role: 'user' },
];

describe('buildChatTransport', () => {
  it('puts messages, id, trigger, and extras into the request body', () => {
    const prepare = getPrepare({
      model: 'claude-sonnet-4-6',
      temperature: 0.3,
    });
    const { body } = prepare({
      id: 'conv-1',
      messageId: 'm-1',
      messages: sampleMessages,
      trigger: SUBMIT,
    });

    expect(body).toMatchObject({
      id: 'conv-1',
      messageId: 'm-1',
      messages: sampleMessages,
      model: 'claude-sonnet-4-6',
      posthogDistinctId: 'browser-distinct-id',
      posthogSessionId: 'session-test-id',
      temperature: 0.3,
      trigger: SUBMIT,
    });
    expect(body).not.toHaveProperty('userId');
  });

  it('forwards all sampling params when present', () => {
    const prepare = getPrepare({
      embeddingsModel: 'BAAI/bge-m3',
      maxTokens: 2_048,
      model: GPT_MODEL,
      queryTransformMode: 'rewrite_hyde',
      queryTransformModel: GPT_MODEL,
      temperature: 0.5,
      topP: 0.9,
    });
    const { body } = prepare({
      id: 'c',
      messages: sampleMessages,
      trigger: SUBMIT,
    });

    expect(body).toMatchObject({
      embeddingsModel: 'BAAI/bge-m3',
      maxTokens: 2_048,
      model: GPT_MODEL,
      queryTransformMode: 'rewrite_hyde',
      queryTransformModel: GPT_MODEL,
      temperature: 0.5,
      topP: 0.9,
    });
  });

  it('reads extras lazily on every call (picks up model changes)', () => {
    let model = 'model-a';
    const transport = buildChatTransport(() => ({
      model,
    })) as unknown as PrepareTransport;
    const first = transport.prepareSendMessagesRequest({
      id: 'c',
      messages: sampleMessages,
      trigger: SUBMIT,
    });

    expect((first.body as { model: string }).model).toBe('model-a');

    model = 'model-b';

    const second = transport.prepareSendMessagesRequest({
      id: 'c',
      messages: sampleMessages,
      trigger: SUBMIT,
    });

    expect((second.body as { model: string }).model).toBe('model-b');
  });

  it('prepares resume requests with only the conversation stream URL', () => {
    const transport = buildChatTransport(() => ({
      model: 'model-a',
    })) as unknown as PrepareTransport;

    const request = transport.prepareReconnectToStreamRequest({ id: 'conv-7' });

    expect(request.api).toBe('/api/chat/conv-7/stream');
    expect(request.headers).toBeUndefined();
  });

  it('calls the explicit stop endpoint without client ownership headers', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null));
    vi.stubGlobal('fetch', fetchMock);

    await stopChatStream('conv-7', {
      activeStreamId: ACTIVE_STREAM_ID,
    });

    expect(fetchMock).toHaveBeenCalledWith('/api/chat/conv-7/stop', {
      body: JSON.stringify({
        activeStreamId: ACTIVE_STREAM_ID,
      }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
  });

  it('throws when the explicit stop endpoint rejects the request', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(null, { status: 500 })),
    );

    await expect(stopChatStream('conv-7')).rejects.toBeInstanceOf(
      StopChatStreamError,
    );
    await expect(stopChatStream('conv-7')).rejects.toMatchObject({
      status: 500,
    });
  });
});

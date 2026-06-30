import { describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { buildChatTransport, type ChatExtras } from '@/lib/transport';

vi.mock('@/lib/user', () => ({
  ANON_USER_ID_KEY: 'finkiHub.anonUserId',
  getAnonUserId: () => 'anon-test-id',
}));

const SUBMIT = 'submit-message' as const;

type PrepareArgs = {
  id: string;
  messageId?: string;
  messages: MyUIMessage[];
  trigger: 'regenerate-message' | 'submit-message';
};

type PrepareTransport = {
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
  it('puts messages, id, trigger, extras, and userId into the request body', () => {
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
      temperature: 0.3,
      trigger: SUBMIT,
      userId: 'anon-test-id',
    });
  });

  it('forwards all sampling params when present', () => {
    const prepare = getPrepare({
      embeddingsModel: 'BAAI/bge-m3',
      maxTokens: 2_048,
      model: 'gpt-5.4-mini',
      queryTransformMode: 'rewrite_hyde',
      queryTransformModel: 'gpt-5.4-mini',
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
      model: 'gpt-5.4-mini',
      queryTransformMode: 'rewrite_hyde',
      queryTransformModel: 'gpt-5.4-mini',
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
});

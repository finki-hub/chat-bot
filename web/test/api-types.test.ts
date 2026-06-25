import { describe, expect, it } from 'vitest';

import {
  type ChatRequestBody,
  type FeedbackClientPayload,
  type FeedbackSchema,
  MAX_CHARS_PER_TURN,
  MAX_MESSAGES,
  type MyUIMessage,
  type ProtocolV2Event,
} from '@/lib/api-types';

describe('lib/api-types', () => {
  it('exposes the wire caps', () => {
    expect(MAX_MESSAGES).toBe(50);
    expect(MAX_CHARS_PER_TURN).toBe(8_000);
  });

  it('models a ChatRequestBody (oldest-first, last is user)', () => {
    const body = {
      messages: [
        { content: 'здраво', role: 'user' },
        { content: 'здраво!', role: 'assistant' },
        { content: 'кога е испитот?', role: 'user' },
      ],
      temperature: 0.3,
    } satisfies ChatRequestBody;

    expect(body.messages.at(-1)?.role).toBe('user');
    expect(body.temperature).toBe(0.3);
  });

  it('models a protocol-v2 token event', () => {
    const ev = {
      data: { text: 'збор' },
      event: 'token',
    } satisfies ProtocolV2Event;

    expect(ev.event).toBe('token');
    expect(ev.data.text).toBe('збор');
  });

  it('models the feedback wire + client payloads', () => {
    const wire = {
      client: 'web',
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      feedback_type: 'like',
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      response_id: '00000000-0000-4000-8000-000000000000',
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      user_id: 'anon-1',
    } satisfies FeedbackSchema;
    const payload = {
      feedbackType: 'like',
      responseId: wire.response_id,
      userId: wire.user_id,
    } satisfies FeedbackClientPayload;

    expect(wire.client).toBe('web');
    expect(wire.feedback_type).toBe('like');
    expect(wire.response_id).toBe('00000000-0000-4000-8000-000000000000');
    expect(payload.feedbackType).toBe('like');
    expect(payload.responseId).toBe(wire.response_id);
    expect(payload.userId).toBe('anon-1');
  });

  it('models a typed UIMessage with metadata + data-status part', () => {
    const msg: MyUIMessage = {
      id: 'm1',
      metadata: { inferenceModel: 'claude-sonnet-4-6', responseId: 'r1' },
      parts: [
        { text: 'одговор', type: 'text' },
        {
          data: { label: '🔍 Пребарувам…', tool: 'search' },
          type: 'data-status',
        },
      ],
      role: 'assistant',
    };

    expect(msg.id).toBe('m1');
    expect(msg.role).toBe('assistant');
    expect(msg.metadata?.inferenceModel).toBe('claude-sonnet-4-6');
    expect(msg.metadata?.responseId).toBe('r1');
    expect(msg.parts[0]?.type).toBe('text');
  });
});

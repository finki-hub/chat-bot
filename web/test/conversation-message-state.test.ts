import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import {
  applyFeedback,
  reconcileHydratedMessages,
  replaceFinishedMessage,
} from '@/lib/conversation-message-state';

const assistantMessage = (
  id: string,
  responseId: string,
  text: string,
): MyUIMessage => ({
  id,
  metadata: { responseId },
  parts: [{ text, type: 'text' }],
  role: 'assistant',
});

describe('replaceFinishedMessage', () => {
  it('upserts a resumed finish by responseId instead of appending a duplicate', () => {
    const previous: MyUIMessage[] = [
      { id: 'u1', parts: [{ text: 'Прашање', type: 'text' }], role: 'user' },
      assistantMessage('a-persisted', 'resp-1', 'old partial'),
      assistantMessage('stream-resumed', 'resp-1', 'new complete'),
    ];

    const next = replaceFinishedMessage({
      pruneAfterReplacement: false,
      replacement: assistantMessage('stream-resumed', 'resp-1', 'new complete'),
      streamMessageId: 'stream-resumed',
    })(previous);

    expect(next.map((message) => message.id)).toStrictEqual([
      'u1',
      'a-persisted',
    ]);
    expect(next.at(1)?.parts).toStrictEqual([
      { text: 'new complete', type: 'text' },
    ]);
    expect(next.at(1)?.metadata?.responseId).toBe('resp-1');
  });
});

describe('reconcileHydratedMessages', () => {
  it('keeps a resumed assistant when persisted history arrives after the stream', () => {
    const persisted: MyUIMessage[] = [
      { id: 'u1', parts: [{ text: 'Прашање', type: 'text' }], role: 'user' },
    ];
    const resumed = assistantMessage('stream-1', 'resp-1', 'Нов одговор');

    const next = reconcileHydratedMessages({
      activeStream: { id: 'resp-1', replacementMessageId: null },
      current: [resumed],
      persisted,
    });

    expect(next.map((message) => message.id)).toStrictEqual(['u1', 'stream-1']);
    expect(next.at(1)?.parts).toStrictEqual([
      { text: 'Нов одговор', type: 'text' },
    ]);
  });

  it('restores a regenerated target and prunes later turns after refresh', () => {
    const persisted: MyUIMessage[] = [
      { id: 'u1', parts: [{ text: 'Прво', type: 'text' }], role: 'user' },
      assistantMessage('a1', 'old-1', 'Стар одговор'),
      { id: 'u2', parts: [{ text: 'Второ', type: 'text' }], role: 'user' },
      assistantMessage('a2', 'old-2', 'Подоцнежен одговор'),
    ];
    const resumed: MyUIMessage = {
      ...assistantMessage('stream-2', 'resp-2', 'Регенериран одговор'),
      metadata: {
        replacementMessageId: 'a1',
        responseId: 'resp-2',
      },
    };

    const next = reconcileHydratedMessages({
      activeStream: { id: 'resp-2', replacementMessageId: 'a1' },
      current: [resumed],
      persisted,
    });

    expect(next.map((message) => message.id)).toStrictEqual(['u1', 'a1']);
    expect(next.at(1)?.parts).toStrictEqual([
      { text: 'Регенериран одговор', type: 'text' },
    ]);
    expect(next.at(1)?.metadata?.responseId).toBe('resp-2');
  });

  it('prepares the regeneration target when history arrives before the resumed stream', () => {
    const persisted: MyUIMessage[] = [
      { id: 'u1', parts: [{ text: 'Прво', type: 'text' }], role: 'user' },
      assistantMessage('a1', 'old-1', 'Стар одговор'),
      { id: 'u2', parts: [{ text: 'Второ', type: 'text' }], role: 'user' },
      assistantMessage('a2', 'old-2', 'Подоцнежен одговор'),
    ];

    const next = reconcileHydratedMessages({
      activeStream: { id: 'resp-2', replacementMessageId: 'a1' },
      current: [],
      persisted,
    });

    expect(next.map((message) => message.id)).toStrictEqual(['u1', 'a1']);
    expect(next.at(1)?.parts).toStrictEqual([]);
    expect(next.at(1)?.metadata).toStrictEqual({});
  });
});

describe('applyFeedback', () => {
  it('removes only the feedback metadata when the vote is cleared', () => {
    const message: MyUIMessage = {
      ...assistantMessage('a1', 'resp-1', 'Одговор'),
      metadata: {
        feedback: 'like',
        inferenceModel: 'claude-sonnet-5',
        responseId: 'resp-1',
      },
    };

    const [updated] = applyFeedback('a1', null)([message]);

    expect(updated?.metadata).toStrictEqual({
      inferenceModel: 'claude-sonnet-5',
      responseId: 'resp-1',
    });
  });
});

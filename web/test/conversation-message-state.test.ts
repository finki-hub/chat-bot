import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { replaceFinishedMessage } from '@/lib/conversation-message-state';

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

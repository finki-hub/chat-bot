import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { reasoningParts, textParts } from '@/lib/message-parts';

const message = (parts: MyUIMessage['parts']): MyUIMessage => ({
  id: 'm1',
  parts,
  role: 'assistant',
});

describe('reasoningParts', () => {
  it('returns only reasoning parts and leaves text to textParts', () => {
    const msg = message([
      { state: 'done', text: 'thought', type: 'reasoning' },
      { text: 'answer', type: 'text' },
    ]);

    expect(reasoningParts(msg).map((p) => p.text)).toStrictEqual(['thought']);
    expect(textParts(msg).map((p) => p.text)).toStrictEqual(['answer']);
  });

  it('returns an empty array when there is no reasoning', () => {
    expect(
      reasoningParts(message([{ text: 'a', type: 'text' }])),
    ).toStrictEqual([]);
  });
});

import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import {
  answerText,
  joinText,
  reasoningParts,
  textParts,
} from '@/lib/message-parts';

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

describe('answerText', () => {
  const toolPart: MyUIMessage['parts'][number] = {
    input: { query: 'рок' },
    output: { summary: '' },
    state: 'output-available',
    toolCallId: 't1',
    toolName: 'search_knowledge_base',
    type: 'dynamic-tool',
  };

  it('joins every segment around tool calls with a blank line', () => {
    const msg = message([
      { text: 'Проверувам.', type: 'text' },
      toolPart,
      { text: '', type: 'text' },
      { text: 'Одговор.', type: 'text' },
    ]);

    expect(answerText(msg)).toBe('Проверувам.\n\nОдговор.');
    expect(joinText(msg)).toBe('Проверувам.\n\nОдговор.');
  });

  it('keeps only the last text part for legacy streams without tool calls', () => {
    const msg = message([
      { text: 'preamble', type: 'text' },
      { text: 'answer', type: 'text' },
    ]);

    expect(answerText(msg)).toBe('answer');
    expect(joinText(msg)).toBe('preambleanswer');
  });
});

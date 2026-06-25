import { describe, expect, it } from 'vitest';

import {
  MAX_CHARS_PER_TURN,
  MAX_MESSAGES,
  type MyUIMessage,
} from '@/lib/api-types';
import { deriveTitle, trimForRequest } from '@/lib/messages';

const textMsg = (
  id: string,
  role: MyUIMessage['role'],
  text: string,
): MyUIMessage => ({
  id,
  parts: [{ text, type: 'text' }],
  role,
});

const textLength = (message: MyUIMessage): number => {
  let total = 0;

  for (const part of message.parts) {
    if (part.type === 'text') {
      total += part.text.length;
    }
  }

  return total;
};

describe('trimForRequest', () => {
  it('keeps the newest MAX_MESSAGES when over the cap', () => {
    const msgs: MyUIMessage[] = Array.from(
      { length: MAX_MESSAGES + 5 },
      (_, i) => textMsg(`m${i}`, i % 2 === 0 ? 'user' : 'assistant', `t${i}`),
    );
    const trimmed = trimForRequest(msgs);

    expect(trimmed).toHaveLength(MAX_MESSAGES);
    expect(trimmed[0]?.id).toBe('m5');
    expect(trimmed.at(-1)?.id).toBe(`m${MAX_MESSAGES + 4}`);
  });

  it('returns the same list when under the cap', () => {
    const msgs = [textMsg('a', 'user', 'hi')];

    expect(trimForRequest(msgs)).toHaveLength(1);
  });

  it('truncates an over-long turn to MAX_CHARS_PER_TURN', () => {
    const long = 'я'.repeat(MAX_CHARS_PER_TURN + 100);
    const trimmed = trimForRequest([textMsg('a', 'user', long)]);
    const part = trimmed[0]?.parts[0];

    expect(part?.type).toBe('text');
    expect(part?.type === 'text' ? part.text : '').toHaveLength(
      MAX_CHARS_PER_TURN,
    );
  });

  it('truncates across multiple text parts by combined length', () => {
    const a = 'a'.repeat(5_000);
    const b = 'b'.repeat(5_000);
    const msg: MyUIMessage = {
      id: 'm',
      parts: [
        { text: a, type: 'text' },
        { text: b, type: 'text' },
      ],
      role: 'user',
    };
    const trimmed = trimForRequest([msg]);
    const total = trimmed[0] ? textLength(trimmed[0]) : 0;

    expect(total).toBe(MAX_CHARS_PER_TURN);
  });
});

describe('deriveTitle', () => {
  it('uses the first line, trimmed', () => {
    expect(deriveTitle('  Која е оценката?\nвтор ред  ')).toBe(
      'Која е оценката?',
    );
  });

  it('truncates to 60 chars with an ellipsis', () => {
    const title = deriveTitle('a'.repeat(100));

    expect(title).toHaveLength(60);
    expect(title.at(-1)).toBe('…');
  });

  it('falls back when empty', () => {
    expect(deriveTitle(' '.repeat(3))).toBe('Нов разговор');
    expect(deriveTitle('')).toBe('Нов разговор');
  });
});

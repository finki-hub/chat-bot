import { describe, expect, it } from 'vitest';

import { deriveTitle } from '@/lib/messages';

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

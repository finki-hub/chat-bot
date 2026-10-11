import { describe, expect, it } from 'vitest';

import { normalizeModelId } from '@/lib/model-aliases';

describe('normalizeModelId', () => {
  it.each([
    ['gpt-5.6-luna', 'gpt-6-luna'],
    ['gpt-5.6-sol', 'gpt-6.1-sol'],
    ['claude-opus-4-8', 'claude-opus-5-5'],
    ['claude-sonnet-5', 'claude-sonnet-5-5'],
    ['claude-haiku-4-5', 'claude-haiku-5-5'],
    ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite'],
    ['openrouter:x-ai/grok-4.6', 'openrouter:x-ai/grok-4.7'],
  ])('maps %s to %s', (legacy, current) => {
    expect(normalizeModelId(legacy)).toBe(current);
  });

  it('does not change unrelated models', () => {
    expect(normalizeModelId('text-embedding-3-small')).toBe(
      'text-embedding-3-small',
    );
    expect(normalizeModelId('toString')).toBe('toString');
    expect(normalizeModelId('constructor')).toBe('constructor');
  });
});

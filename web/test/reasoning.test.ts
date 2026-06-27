import { describe, expect, it } from 'vitest';

import { isReasoningCapableModel } from '@/lib/reasoning';

describe('isReasoningCapableModel', () => {
  it('is true for reasoning-capable model families', () => {
    const capable = [
      'claude-sonnet-4-6',
      'claude-opus-4-8',
      'gemini-2.5-flash',
      'gemini-3-flash-preview',
      'gpt-5.4-mini',
      'deepseek-r1:70b',
    ];

    expect(capable.map(isReasoningCapableModel)).toStrictEqual(
      capable.map(() => true),
    );
  });

  it('is false for non-reasoning models', () => {
    const incapable = ['gpt-4.1', 'gpt-4o-mini', 'llama3.3:70b', 'BAAI/bge-m3'];

    expect(incapable.map(isReasoningCapableModel)).toStrictEqual(
      incapable.map(() => false),
    );
  });
});

import { describe, expect, it } from 'vitest';

import { isReasoningCapableModel } from '@/lib/reasoning';

describe('isReasoningCapableModel', () => {
  it('is true for reasoning-capable model families', () => {
    const capable = [
      'claude-sonnet-5',
      'claude-opus-4-8',
      'gemini-3.5-flash',
      'claude-haiku-4-5',
      'gpt-5.4-mini',
      'gpt-5.6-sol',
      'qwen3:30b-a3b-thinking-2507-q4_K_M',
      'qwen3:14b-q4_K_M',
    ];

    expect(capable.map(isReasoningCapableModel)).toStrictEqual(
      capable.map(() => true),
    );
  });

  it('is false for non-reasoning models', () => {
    const incapable = ['qwen3:30b-a3b-instruct-2507-q4_K_M', 'BAAI/bge-m3'];

    expect(incapable.map(isReasoningCapableModel)).toStrictEqual(
      incapable.map(() => false),
    );
  });
});

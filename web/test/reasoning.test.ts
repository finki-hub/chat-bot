import { describe, expect, it } from 'vitest';

import { isReasoningCapableModel } from '@/lib/reasoning';

describe('isReasoningCapableModel', () => {
  it('is true for reasoning-capable model families', () => {
    const capable = [
      'claude-sonnet-5',
      'claude-opus-4-8',
      'gemini-2.5-flash',
      'claude-haiku-4-5',
      'gpt-5.4-mini',
      'deepseek-r1:70b',
    ];

    expect(capable.map(isReasoningCapableModel)).toStrictEqual(
      capable.map(() => true),
    );
  });

  it('is false for non-reasoning models', () => {
    const incapable = [
      'llama3.3:70b',
      'hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0',
      'hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0',
      'BAAI/bge-m3',
    ];

    expect(incapable.map(isReasoningCapableModel)).toStrictEqual(
      incapable.map(() => false),
    );
  });
});

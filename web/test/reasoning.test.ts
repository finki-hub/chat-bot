import { describe, expect, it } from 'vitest';

import {
  isReasoningCapableModel,
  isReasoningEnabledForModel,
  isReasoningMandatoryModel,
} from '@/lib/reasoning';

describe('isReasoningCapableModel', () => {
  it('is true for reasoning-capable model families', () => {
    const capable = [
      'claude-sonnet-5',
      'claude-opus-4-8',
      'gemini-3.5-flash',
      'claude-haiku-4-5',
      'gpt-5.4-mini',
      'gpt-5.6-sol',
      'openrouter:deepseek/deepseek-v4-pro-0813',
      'openrouter:moonshotai/kimi-k3',
      'openrouter:qwen/qwen3.8-max',
      'openrouter:x-ai/grok-4.6',
      'openrouter:tencent/hy3',
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

describe('mandatory reasoning policy', () => {
  const mandatoryModel = 'openrouter:z-ai/glm-5.3';

  it('identifies only models whose provider cannot disable reasoning', () => {
    expect(isReasoningMandatoryModel(mandatoryModel)).toBe(true);
    expect(
      isReasoningMandatoryModel('openrouter:deepseek/deepseek-v4-pro-0813'),
    ).toBe(false);
  });

  it('keeps mandatory reasoning enabled when the stored preference is false', () => {
    expect(isReasoningEnabledForModel(mandatoryModel, false)).toBe(true);
    expect(
      isReasoningEnabledForModel(
        'openrouter:deepseek/deepseek-v4-pro-0813',
        false,
      ),
    ).toBe(false);
  });
});

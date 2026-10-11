import { beforeEach, describe, expect, it } from 'vitest';

import { DEFAULT_MODEL, useUiStore } from '@/lib/ui-store';

const STORAGE_KEY = 'finkiHub.ui';
const LEGACY_MAPPINGS = [
  ['gpt-5.4', 'gpt-5.6-terra'],
  ['gpt-5.5', 'gpt-6.1-sol'],
  ['gemini-3.5-flash', 'gemini-3.8-flash'],
  [
    'openrouter:deepseek/deepseek-v4-flash',
    'openrouter:deepseek/deepseek-v4.1-flash',
  ],
  [
    'openrouter:deepseek/deepseek-v4-flash-0731',
    'openrouter:deepseek/deepseek-v4.1-flash',
  ],
  ['openrouter:qwen/qwen3.8-max', 'openrouter:qwen/qwen3.8-max-0902'],
  ['gpt-5.6-luna', 'gpt-6-luna'],
  ['gpt-5.6-sol', 'gpt-6.1-sol'],
  ['claude-opus-4-8', 'claude-opus-5-5'],
  ['claude-sonnet-5', 'claude-sonnet-5-5'],
  ['claude-haiku-4-5', 'claude-haiku-5-5'],
  ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite'],
  ['openrouter:x-ai/grok-4.6', 'openrouter:x-ai/grok-4.7'],
] as const;

const storeValue = (): { state: Record<string, unknown>; version: number } =>
  JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as {
    state: Record<string, unknown>;
    version: number;
  };

describe('versioned persisted model selection', () => {
  beforeEach(() => {
    localStorage.clear();
    useUiStore.setState({
      activeConversationId: null,
      model: DEFAULT_MODEL,
      reasoning: false,
    });
  });

  it('keeps exactly thirteen one-time migration entries', () => {
    expect(LEGACY_MAPPINGS).toHaveLength(13);
  });

  it.each(LEGACY_MAPPINGS)(
    'migrates version 0 %s to %s once and persists version 1',
    async (legacyModel, currentModel) => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          state: {
            activeConversationId: 'saved-conversation',
            model: legacyModel,
            reasoning: true,
          },
          version: 0,
        }),
      );

      await useUiStore.persist.rehydrate();

      expect(useUiStore.getState()).toMatchObject({
        activeConversationId: 'saved-conversation',
        model: currentModel,
        reasoning: true,
      });
      expect(storeValue()).toStrictEqual({
        state: {
          activeConversationId: 'saved-conversation',
          model: currentModel,
          reasoning: true,
        },
        version: 1,
      });
    },
  );

  it('preserves unknown model preferences during version 0 migration', async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ state: { model: 'unknown-model-id' }, version: 0 }),
    );

    await useUiStore.persist.rehydrate();

    expect(useUiStore.getState().model).toBe('unknown-model-id');
    expect(storeValue()).toMatchObject({
      state: { model: 'unknown-model-id' },
      version: 1,
    });
  });

  it('does not rerun migration for version 1 persisted data', async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ state: { model: 'claude-sonnet-5' }, version: 1 }),
    );

    await useUiStore.persist.rehydrate();

    expect(useUiStore.getState().model).toBe('claude-sonnet-5');
    expect(storeValue()).toStrictEqual({
      state: { model: 'claude-sonnet-5' },
      version: 1,
    });
  });

  it('ignores malformed persisted fields and preserves the safe current defaults', async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          activeConversationId: [],
          model: { id: 'gpt-5.6-luna' },
          reasoning: 'yes',
        },
        version: 0,
      }),
    );

    await useUiStore.persist.rehydrate();

    expect(useUiStore.getState()).toMatchObject({
      activeConversationId: null,
      model: DEFAULT_MODEL,
      reasoning: false,
    });
  });

  it('keeps model selections verbatim outside the one-time migration', () => {
    useUiStore.getState().setModel('gpt-5.6-sol');

    expect(useUiStore.getState().model).toBe('gpt-5.6-sol');
    expect(storeValue()).toMatchObject({
      state: { model: 'gpt-5.6-sol' },
      version: 1,
    });
  });
});

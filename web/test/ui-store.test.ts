import { beforeEach, describe, expect, it } from 'vitest';

import { DEFAULT_MODEL, useUiStore } from '@/lib/ui-store';

describe('persisted model selection', () => {
  beforeEach(() => {
    localStorage.clear();
    useUiStore.setState({ model: DEFAULT_MODEL });
  });

  it('normalizes a legacy model while hydrating localStorage', async () => {
    localStorage.setItem(
      'finkiHub.ui',
      JSON.stringify({ state: { model: 'claude-sonnet-5' }, version: 0 }),
    );

    await useUiStore.persist.rehydrate();

    expect(useUiStore.getState().model).toBe('claude-sonnet-5-5');
  });

  it('keeps catalog IDs passed to setModel intact for recovery', () => {
    useUiStore.getState().setModel('gpt-5.6-sol');

    expect(useUiStore.getState().model).toBe('gpt-5.6-sol');
    expect(
      JSON.parse(localStorage.getItem('finkiHub.ui') ?? '{}'),
    ).toMatchObject({
      state: { model: 'gpt-6.1-sol' },
    });
  });
});

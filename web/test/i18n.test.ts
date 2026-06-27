import { describe, expect, it } from 'vitest';

import { formatSpanLabel, messages, t } from '@/lib/i18n';

describe('i18n', () => {
  it('returns the Macedonian string for a known key', () => {
    expect(t('sidebar.new')).toBe('Нов разговор');
    expect(t('thread.emptyTitle')).toBe('Започни разговор');
    expect(t('error.retry')).toBe('Обиди се повторно');
    expect(t('error.interrupted')).toBe('Одговорот е прекинат.');
  });

  it('exposes the flat dictionary', () => {
    expect(messages['actions.copy']).toBe('Копирај');
    expect(messages['app.title']).toBe('ФИНКИ Хаб');
  });

  it('covers every documented key with non-empty values', () => {
    const keys = Object.keys(messages);

    expect(keys.length).toBeGreaterThanOrEqual(20);

    for (const key of keys) {
      expect(messages[key as keyof typeof messages].length).toBeGreaterThan(0);
    }
  });

  it('translates known span keys and falls back to the raw key otherwise', () => {
    expect(formatSpanLabel('retrieval.embed')).toBe('вградување');
    expect(formatSpanLabel('agent.setup')).toBe('агент: подготовка');
    expect(formatSpanLabel('retrieval.future_stage')).toBe(
      'retrieval.future_stage',
    );
  });
});

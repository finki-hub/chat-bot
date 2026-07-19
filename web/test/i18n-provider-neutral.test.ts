import { describe, expect, it } from 'vitest';

import { messages, t } from '@/lib/i18n';

describe('provider-neutral credential copy', () => {
  it('keeps the credential button generic and removes the sponsored-only key', () => {
    expect(t('error.manageCredentials')).toBe('Додај API клуч');
    expect('error.sponsoredManageCredentials' in messages).toBe(false);
  });
});

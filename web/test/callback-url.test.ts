import { describe, expect, it } from 'vitest';

import { getSafeCallbackUrl } from '@/lib/callback-url';

describe('getSafeCallbackUrl', () => {
  it('keeps same-origin relative callback URLs', () => {
    expect(getSafeCallbackUrl('/chat/abc?source=signin')).toBe(
      '/chat/abc?source=signin',
    );
  });

  it('falls back for missing callback URLs', () => {
    expect(getSafeCallbackUrl(undefined)).toBe('/');
  });

  it('falls back for absolute external callback URLs', () => {
    expect(getSafeCallbackUrl('https://evil.example/phish')).toBe('/');
  });

  it('falls back for protocol-relative callback URLs', () => {
    expect(getSafeCallbackUrl('//evil.example/phish')).toBe('/');
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const ORIGINAL = { ...process.env };
const API_BASE_URL_PATTERN = /API_BASE_URL/u;
const CHAT_API_KEY_PATTERN = /CHAT_API_KEY/u;

const withoutKey = (
  source: typeof process.env,
  key: string,
): typeof process.env =>
  Object.fromEntries(
    Object.entries(source).filter(([name]) => name !== key),
  ) as typeof process.env;

describe('lib/env', () => {
  beforeEach(() => {
    vi.resetModules();
    process.env = { ...ORIGINAL };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL };
  });

  it('throws when API_BASE_URL is missing', async () => {
    process.env = withoutKey(process.env, 'API_BASE_URL');
    process.env['CHAT_API_KEY'] = 'k';

    await expect(import('@/lib/env')).rejects.toThrow(API_BASE_URL_PATTERN);
  });

  it('throws when CHAT_API_KEY is missing', async () => {
    process.env = withoutKey(process.env, 'CHAT_API_KEY');
    process.env['API_BASE_URL'] = 'https://api:8880';

    await expect(import('@/lib/env')).rejects.toThrow(CHAT_API_KEY_PATTERN);
  });

  it('exposes both values when set', async () => {
    process.env['API_BASE_URL'] = 'https://api:8880';
    process.env['CHAT_API_KEY'] = 'secret-key';
    const mod = await import('@/lib/env');

    expect(mod.API_BASE_URL).toBe('https://api:8880');
    expect(mod.CHAT_API_KEY).toBe('secret-key');
    expect(mod.env).toStrictEqual({
      API_BASE_URL: 'https://api:8880',
      CHAT_API_KEY: 'secret-key',
    });
  });
});

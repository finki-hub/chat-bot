import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const ORIGINAL = { ...process.env };
const API_BASE_URL_VALUE = 'https://api:8880';
const API_BASE_URL_PATTERN = /API_BASE_URL/u;
const CHAT_API_KEY_PATTERN = /CHAT_API_KEY/u;
const RESUMABLE_STREAM_REDIS_URL_PATTERN = /RESUMABLE_STREAM_REDIS_URL/u;

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
    process.env['API_BASE_URL'] = API_BASE_URL_VALUE;
    process.env['RESUMABLE_STREAM_REDIS_URL'] = 'redis://stream-store:6379';

    await expect(import('@/lib/env')).rejects.toThrow(CHAT_API_KEY_PATTERN);
  });

  it('throws a server-only config error when RESUMABLE_STREAM_REDIS_URL is missing', async () => {
    process.env = withoutKey(process.env, 'RESUMABLE_STREAM_REDIS_URL');
    process.env['API_BASE_URL'] = API_BASE_URL_VALUE;
    process.env['CHAT_API_KEY'] = 'secret-key';

    await expect(import('@/lib/env')).rejects.toThrow(
      RESUMABLE_STREAM_REDIS_URL_PATTERN,
    );
  });

  it('exposes both values when set', async () => {
    process.env['API_BASE_URL'] = API_BASE_URL_VALUE;
    process.env['CHAT_API_KEY'] = 'secret-key';
    process.env['RESUMABLE_STREAM_REDIS_URL'] = 'redis://stream-store:6379';
    const mod = await import('@/lib/env');

    expect(mod.API_BASE_URL).toBe(API_BASE_URL_VALUE);
    expect(mod.CHAT_API_KEY).toBe('secret-key');
    expect(mod.RESUMABLE_STREAM_REDIS_URL).toBe('redis://stream-store:6379');
    expect(mod.env).toStrictEqual({
      API_BASE_URL: API_BASE_URL_VALUE,
      CHAT_API_KEY: 'secret-key',
      RESUMABLE_STREAM_REDIS_URL: 'redis://stream-store:6379',
    });
  });
});

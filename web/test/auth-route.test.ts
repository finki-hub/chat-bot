import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const ORIGINAL = { ...process.env };
const AUTH_GOOGLE_ID_PATTERN = /AUTH_GOOGLE_ID/u;

vi.mock('next-auth', () => ({
  default: vi.fn<
    () => {
      readonly auth: () => void;
      readonly handlers: {
        readonly GET: () => void;
        readonly POST: () => void;
      };
      readonly signIn: () => void;
      readonly signOut: () => void;
    }
  >(() => ({
    auth: vi.fn<() => void>(),
    handlers: { GET: vi.fn<() => void>(), POST: vi.fn<() => void>() },
    signIn: vi.fn<() => void>(),
    signOut: vi.fn<() => void>(),
  })),
}));

vi.mock('next-auth/providers/google', () => ({
  default: vi.fn<
    (config: unknown) => { readonly config: unknown; readonly id: 'google' }
  >((config) => ({ config, id: 'google' })),
}));

const setAuthEnv = (): void => {
  process.env['AUTH_GOOGLE_ID'] = 'google-client-id';
  process.env['AUTH_GOOGLE_SECRET'] = 'google-client-secret';
  process.env['AUTH_SECRET'] = 'auth-secret';
  process.env['AUTH_URL'] = 'http://localhost:3000';
};

describe('Auth.js route handler', () => {
  beforeEach(() => {
    vi.resetModules();
    process.env = { ...ORIGINAL };
    setAuthEnv();
  });

  afterEach(() => {
    process.env = { ...ORIGINAL };
  });

  it('exports GET and POST handlers for the App Router catch-all route', async () => {
    const route = await import('@/app/api/auth/[...nextauth]/route');

    expect(route.GET).toBeTypeOf('function');
    expect(route.POST).toBeTypeOf('function');
  });

  it('fails fast when the Google client id is missing', async () => {
    process.env['AUTH_GOOGLE_ID'] = '';

    await expect(import('@/auth')).rejects.toThrow(AUTH_GOOGLE_ID_PATTERN);
  });
});

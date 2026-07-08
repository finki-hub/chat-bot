import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const ORIGINAL = { ...process.env };
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

vi.mock('next-auth/providers/microsoft-entra-id', () => ({
  default: vi.fn<
    (config: unknown) => {
      readonly config: unknown;
      readonly id: 'microsoft-entra-id';
    }
  >((config) => ({ config, id: 'microsoft-entra-id' })),
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

  it('does not throw at module import when runtime auth env is missing', async () => {
    process.env['AUTH_GOOGLE_ID'] = '';

    await expect(import('@/auth')).resolves.toHaveProperty('auth');
  });

  it('is configured when only Microsoft Entra ID credentials are present', async () => {
    process.env['AUTH_GOOGLE_ID'] = '';
    process.env['AUTH_GOOGLE_SECRET'] = '';
    process.env['AUTH_MICROSOFT_ENTRA_ID_ID'] = 'microsoft-client-id';
    process.env['AUTH_MICROSOFT_ENTRA_ID_SECRET'] = 'microsoft-client-secret';
    process.env['AUTH_MICROSOFT_ENTRA_ID_ISSUER'] =
      'https://login.microsoftonline.com/tenant-id/v2.0/';

    const { isAuthConfigured } = await import('@/auth');

    expect(isAuthConfigured()).toBe(true);
  });

  it('requires Auth.js secret and at least one OAuth provider', async () => {
    process.env['AUTH_GOOGLE_ID'] = '';
    process.env['AUTH_GOOGLE_SECRET'] = '';

    const { isAuthConfigured } = await import('@/auth');

    expect(isAuthConfigured()).toBe(false);
  });
});

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const {
  authMock,
  authState,
  getSafeCallbackUrlMock,
  redirectMock,
  signInMock,
} = vi.hoisted(() => ({
  authMock: vi.fn<() => Promise<null>>(),
  authState: {
    providerMap: [{ id: 'google' as const, name: 'Google' }],
  },
  getSafeCallbackUrlMock: vi.fn<(callbackUrl?: string) => string>(() => '/'),
  redirectMock: vi.fn<(url: string) => never>((url) => {
    throw new Error(`redirect:${url}`);
  }),
  signInMock: vi.fn<() => Promise<void>>(),
}));

vi.mock('@/auth', () => ({
  auth: authMock,
  isAuthConfigured: () => true,
  get providerMap() {
    return authState.providerMap;
  },
  signIn: signInMock,
}));

vi.mock('@/lib/callback-url', () => ({
  getSafeCallbackUrl: getSafeCallbackUrlMock,
}));

vi.mock('next-auth', () => ({
  AuthError: class AuthError extends Error {},
}));

vi.mock('next/navigation', () => ({
  redirect: redirectMock,
}));

const defaultSearchParams = Promise.resolve({});
const oauthErrorSearchParams = Promise.resolve({ error: 'OAuthSignin' });

const renderSignInPage = async (
  searchParams = defaultSearchParams,
): Promise<void> => {
  const { default: SignInPage } = await import('@/app/signin/page');
  const result = await SignInPage({ searchParams });

  render(result);
};

describe('SignInPage', () => {
  beforeEach(() => {
    vi.resetModules();
    authMock.mockReset();
    authMock.mockResolvedValue(null);
    getSafeCallbackUrlMock.mockClear();
    authState.providerMap = [{ id: 'google', name: 'Google' }];
    redirectMock.mockClear();
    signInMock.mockReset();
    signInMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    cleanup();
  });

  it('presents the student-first value and provider action', async () => {
    await renderSignInPage();

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
      'Одговори за ФИНКИ, кога ти требаат.',
    );
    expect(
      screen.getByRole('button', { name: 'Продолжи со Google' }),
    ).toBeVisible();
  });

  it('announces an OAuth failure with a recovery step', async () => {
    await renderSignInPage(oauthErrorSearchParams);

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Не успеавме да те најавиме. Обиди се повторно.',
    );
  });

  it('explains temporary unavailability without server terminology', async () => {
    authState.providerMap = [];

    await renderSignInPage();

    expect(
      screen.getByText(
        'Најавувањето моментално не е достапно. Обиди се повторно подоцна.',
      ),
    ).toBeVisible();
  });
});

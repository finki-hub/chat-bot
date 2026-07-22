import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const {
  authMock,
  authState,
  clientSignInMock,
  getSafeCallbackUrlMock,
  redirectMock,
} = vi.hoisted(() => ({
  authMock: vi.fn<() => Promise<null>>(),
  authState: {
    providerMap: [{ id: 'google' as const, name: 'Google' }],
  },
  clientSignInMock: vi.fn<() => Promise<void>>(),
  getSafeCallbackUrlMock: vi.fn<(callbackUrl?: string) => string>(() => '/'),
  redirectMock: vi.fn<(url: string) => never>((url) => {
    throw new Error(`redirect:${url}`);
  }),
}));

vi.mock('@/auth', () => ({
  auth: authMock,
  isAuthConfigured: () => true,
  get providerMap() {
    return authState.providerMap;
  },
}));

vi.mock('@/lib/callback-url', () => ({
  getSafeCallbackUrl: getSafeCallbackUrlMock,
}));

vi.mock('next-auth/react', () => ({
  signIn: clientSignInMock,
}));

vi.mock('next/navigation', () => ({
  redirect: redirectMock,
}));

const defaultSearchParams = Promise.resolve({});
const oauthErrorSearchParams = Promise.resolve({ error: 'OAuthSignin' });
const GOOGLE_SIGN_IN_LABEL = 'Продолжи со Google';

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
    clientSignInMock.mockReset();
    clientSignInMock.mockResolvedValue(undefined);
    getSafeCallbackUrlMock.mockClear();
    authState.providerMap = [{ id: 'google', name: 'Google' }];
    redirectMock.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it('presents the student-first value and provider action', async () => {
    await renderSignInPage();

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
      'Имаш прашање за ФИНКИ?',
    );
    expect(
      screen.getByRole('button', { name: GOOGLE_SIGN_IN_LABEL }),
    ).toBeVisible();
  });

  it('starts OAuth sign-in from the current browser build', async () => {
    // Given
    const user = userEvent.setup();
    getSafeCallbackUrlMock.mockReturnValue('/chat');
    await renderSignInPage(Promise.resolve({ callbackUrl: '/chat' }));

    // When
    await user.click(
      screen.getByRole('button', { name: GOOGLE_SIGN_IN_LABEL }),
    );

    // Then
    expect(clientSignInMock).toHaveBeenCalledWith('google', {
      redirectTo: '/chat',
    });
  });

  it('starts only one OAuth handshake for repeated clicks', async () => {
    // Given
    const user = userEvent.setup();
    clientSignInMock.mockReturnValue(new Promise<void>(() => {}));
    await renderSignInPage();
    const providerButton = screen.getByRole('button', {
      name: GOOGLE_SIGN_IN_LABEL,
    });

    // When
    await user.dblClick(providerButton);

    // Then
    expect(clientSignInMock).toHaveBeenCalledOnce();
    expect(providerButton).toBeDisabled();
    expect(providerButton).toHaveAttribute('aria-busy', 'true');
  });

  it('allows retry after OAuth initiation rejects', async () => {
    // Given
    const user = userEvent.setup();
    const signInError = new Error('network unavailable');
    clientSignInMock.mockRejectedValueOnce(signInError);
    await renderSignInPage();
    const providerButton = screen.getByRole('button', {
      name: GOOGLE_SIGN_IN_LABEL,
    });

    // When
    await user.click(providerButton);

    // Then
    await waitFor(() => {
      expect(providerButton).toBeEnabled();
    });

    expect(reportError).toHaveBeenCalledWith(signInError);
  });

  it('limits provider arrow movement to motion-safe preferences', async () => {
    await renderSignInPage();

    const providerButton = screen.getByRole('button', {
      name: GOOGLE_SIGN_IN_LABEL,
    });
    const arrow = providerButton.querySelector('.lucide-arrow-right');

    expect(arrow).toHaveClass(
      'motion-safe:transition-[color,transform]',
      'motion-safe:group-hover:translate-x-0.5',
    );
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

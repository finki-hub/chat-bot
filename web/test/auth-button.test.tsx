import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { AuthButton } from '@/components/shell/auth-button';

type MockSessionState =
  | { readonly data: null; readonly status: 'unauthenticated' }
  | {
      readonly data: {
        readonly user: {
          readonly email?: null | string;
          readonly name?: null | string;
        };
      };
      readonly status: 'authenticated';
    };

const authMocks = vi.hoisted(() => ({
  signIn: vi.fn<(provider?: string) => Promise<void>>(() => Promise.resolve()),
  signOut: vi.fn<() => Promise<void>>(() => Promise.resolve()),
  useSession: vi.fn<() => MockSessionState>(),
}));

vi.mock('next-auth/react', () => authMocks);

describe('AuthButton', () => {
  it('opens provider-neutral sign-in when the visitor is unauthenticated', () => {
    authMocks.useSession.mockReturnValue({
      data: null,
      status: 'unauthenticated',
    });

    render(<AuthButton />);
    fireEvent.click(screen.getByRole('button', { name: 'Најави се' }));

    expect(authMocks.signIn).toHaveBeenCalledWith();
  });

  it('signs the current user out when authenticated', () => {
    authMocks.useSession.mockReturnValue({
      data: { user: { email: 'user@example.com', name: 'Test User' } },
      status: 'authenticated',
    });

    render(<AuthButton />);
    const signOutButton = screen.getByRole('button', {
      name: 'Одјави се: Test User',
    });
    fireEvent.click(signOutButton);

    expect(signOutButton).toHaveTextContent('Test User');
    expect(authMocks.signOut).toHaveBeenCalledOnce();
  });

  it('uses the plain sign-out label when authenticated identity is unavailable', () => {
    authMocks.useSession.mockReturnValue({
      data: { user: {} },
      status: 'authenticated',
    });

    render(<AuthButton />);

    const signOutButton = screen.getByRole('button', { name: 'Одјави се' });

    expect(signOutButton).toHaveTextContent('Одјави се');
  });
});

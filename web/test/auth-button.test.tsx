import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { AuthButton } from '@/components/shell/auth-button';

type MockSessionState =
  | { readonly data: null; readonly status: 'unauthenticated' }
  | {
      readonly data: {
        readonly user: { readonly email: string; readonly name: string };
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
    fireEvent.click(screen.getByRole('button', { name: 'Одјави се' }));

    expect(authMocks.signOut).toHaveBeenCalledOnce();
  });
});

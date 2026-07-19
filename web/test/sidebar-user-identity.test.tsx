import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SidebarUserIdentity } from '@/components/shell/sidebar-user-identity';

type MockSessionState = {
  readonly data: {
    readonly user: {
      readonly email: string;
      readonly name: string;
    };
  };
  readonly status: 'authenticated';
};

const authMocks = vi.hoisted(() => ({
  signOut: vi.fn<() => Promise<void>>(() => Promise.resolve()),
  useSession: vi.fn<() => MockSessionState>(),
}));

vi.mock('next-auth/react', () => authMocks);

const USER_EMAIL = 'user@example.com';
const USER_NAME = 'Test User';
const ACCOUNT_MENU_NAME = `Корисничко мени: ${USER_NAME}, ${USER_EMAIL}`;
const CLEAR_ALL_LABEL = 'Избриши ги сите разговори';

describe('SidebarUserIdentity clear-all action', () => {
  beforeEach(() => {
    authMocks.useSession.mockReturnValue({
      data: { user: { email: USER_EMAIL, name: USER_NAME } },
      status: 'authenticated',
    });
  });

  it('requires confirmation before clearing all conversations from the account menu', async () => {
    const onClearAll = vi.fn<() => void>();
    const user = userEvent.setup();

    render(
      <SidebarUserIdentity
        hasConversations
        onClearAll={onClearAll}
        onOpenCredentialsAction={vi.fn<() => void>()}
      />,
    );

    expect(
      screen.queryByRole('menuitem', { name: CLEAR_ALL_LABEL }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: ACCOUNT_MENU_NAME }));
    const clearAllItem = screen.getByRole('menuitem', {
      name: CLEAR_ALL_LABEL,
    });

    expect(clearAllItem).toHaveClass('min-h-11');

    await user.click(clearAllItem);

    expect(onClearAll).not.toHaveBeenCalled();

    const dialog = screen.getByRole('dialog', {
      name: `${CLEAR_ALL_LABEL}?`,
    });

    await user.click(within(dialog).getByTestId('confirm-action'));

    expect(onClearAll).toHaveBeenCalledOnce();
  });

  it('hides the clear-all action when there are no conversations', async () => {
    const user = userEvent.setup();

    render(
      <SidebarUserIdentity
        hasConversations={false}
        onClearAll={vi.fn<() => void>()}
        onOpenCredentialsAction={vi.fn<() => void>()}
      />,
    );

    await user.click(screen.getByRole('button', { name: ACCOUNT_MENU_NAME }));

    expect(
      screen.queryByRole('menuitem', { name: CLEAR_ALL_LABEL }),
    ).not.toBeInTheDocument();
  });
});

import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { Sidebar } from '@/components/shell/sidebar';
import { SidebarUserIdentity } from '@/components/shell/sidebar-user-identity';

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
  signOut: vi.fn<() => Promise<void>>(() => Promise.resolve()),
  useSession: vi.fn<() => MockSessionState>(),
}));

vi.mock('next-auth/react', () => authMocks);

const baseProps = {
  activeId: null,
  conversations: [],
  onClearAll: vi.fn<() => void>(),
  onClose: vi.fn<() => void>(),
  onDelete: vi.fn<(id: string) => void>(),
  onNewChat: vi.fn<() => void>(),
  onRename: vi.fn<(id: string, title: string) => void>(),
  onSelect: vi.fn<(id: string) => void>(),
  open: true,
};

describe('Sidebar conversation loading', () => {
  beforeEach(() => {
    authMocks.signOut.mockClear();
    authMocks.useSession.mockReturnValue({
      data: null,
      status: 'unauthenticated',
    });
  });

  it('uses modal dialog semantics for the mobile drawer', () => {
    render(
      <Sidebar
        {...baseProps}
        footer={<SidebarUserIdentity onOpenCredentials={vi.fn<() => void>()} />}
        mobile
      />,
    );

    expect(
      screen.getByRole('dialog', { name: 'Странична лента' }),
    ).toBeInTheDocument();
  });

  it('opens account actions from the authenticated mobile drawer footer', async () => {
    const onOpenCredentials = vi.fn<() => void>();
    const user = userEvent.setup();
    authMocks.useSession.mockReturnValue({
      data: { user: { email: 'user@example.com', name: 'Test User' } },
      status: 'authenticated',
    });

    render(
      <Sidebar
        {...baseProps}
        footer={<SidebarUserIdentity onOpenCredentials={onOpenCredentials} />}
        mobile
      />,
    );

    const drawer = screen.getByRole('dialog', { name: 'Странична лента' });
    const identity = within(drawer).getByTestId('sidebar-user-identity');

    expect(identity).toHaveTextContent('Test User');
    expect(identity).toHaveTextContent('user@example.com');

    await user.click(
      within(drawer).getByRole('button', {
        name: 'Корисничко мени: Test User, user@example.com',
      }),
    );
    await user.click(screen.getByRole('menuitem', { name: 'API клучеви' }));

    expect(onOpenCredentials).toHaveBeenCalledOnce();

    await user.click(
      within(drawer).getByRole('button', {
        name: 'Корисничко мени: Test User, user@example.com',
      }),
    );
    await user.click(screen.getByRole('menuitem', { name: 'Одјави се' }));

    expect(authMocks.signOut).toHaveBeenCalledOnce();
  });

  it('uses email as the sidebar identity fallback', () => {
    authMocks.useSession.mockReturnValue({
      data: { user: { email: 'user@example.com', name: null } },
      status: 'authenticated',
    });

    render(
      <Sidebar
        {...baseProps}
        footer={<SidebarUserIdentity onOpenCredentials={vi.fn<() => void>()} />}
      />,
    );

    expect(screen.getByTestId('sidebar-user-identity')).toHaveTextContent(
      'user@example.com',
    );
  });

  it('keeps account actions available when authenticated identity is unavailable', async () => {
    const user = userEvent.setup();
    authMocks.useSession.mockReturnValue({
      data: { user: {} },
      status: 'authenticated',
    });
    render(
      <Sidebar
        {...baseProps}
        footer={<SidebarUserIdentity onOpenCredentials={vi.fn<() => void>()} />}
      />,
    );

    const accountTrigger = screen.getByRole('button', {
      name: 'Корисничко мени: Сметка',
    });

    expect(accountTrigger).toHaveTextContent('Сметка');

    await user.click(accountTrigger);

    expect(screen.getByRole('menuitem', { name: 'API клучеви' })).toBeVisible();
    expect(screen.getByRole('menuitem', { name: 'Одјави се' })).toBeVisible();
  });

  it('shows a recoverable error when conversation history cannot load', () => {
    const onRetry = vi.fn<() => Promise<void>>();
    render(
      <Sidebar
        {...baseProps}
        listError
        onRetryList={onRetry}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Разговорите не можеа да се вчитаат.',
    );

    fireEvent.click(screen.getByRole('button', { name: 'Обиди се повторно' }));

    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('does not show a retry action without a retry handler', () => {
    render(
      <Sidebar
        {...baseProps}
        listError
      />,
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Обиди се повторно' }),
    ).not.toBeInTheDocument();
  });

  it('announces initial conversation loading', () => {
    render(
      <Sidebar
        {...baseProps}
        listLoading
        onRetryList={vi.fn<() => Promise<void>>()}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent(
      'Се вчитуваат разговори…',
    );
  });
});

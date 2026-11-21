import type { MouseEvent, ReactNode } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Sidebar } from '@/components/shell/sidebar';
import { t } from '@/lib/i18n';

const closeSidebarOnMobileMock = vi.fn<(onClose: () => void) => void>();

vi.mock('next/link', () => ({
  default: ({
    children,
    href,
    onClick,
    ...props
  }: {
    readonly children: ReactNode;
    readonly href: string;
    readonly onClick?: (event: MouseEvent<HTMLAnchorElement>) => void;
  }) => (
    <a
      href={href}
      onClick={(event) => {
        event.preventDefault();
        onClick?.(event);
      }}
      {...props}
    >
      {children}
    </a>
  ),
}));

vi.mock('@/components/shell/sidebar-helpers', async () => {
  const actual = await import('@/components/shell/sidebar-helpers');
  return {
    ...actual,
    closeSidebarOnMobile: (onClose: () => void) => {
      closeSidebarOnMobileMock(onClose);
    },
  };
});

const createMediaQueryList = (): MediaQueryList => ({
  addEventListener: vi.fn<() => void>(),
  addListener: vi.fn<() => void>(),
  dispatchEvent: vi.fn<() => boolean>(() => true),
  matches: true,
  media: '',
  onchange: null,
  removeEventListener: vi.fn<() => void>(),
  removeListener: vi.fn<() => void>(),
});

describe('Sidebar committee link', () => {
  beforeEach(() => {
    closeSidebarOnMobileMock.mockClear();
    vi.stubGlobal(
      'matchMedia',
      vi.fn<(query: string) => MediaQueryList>(() => createMediaQueryList()),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders a discoverable committee planner link below the new chat button', () => {
    render(
      <Sidebar
        activeId={null}
        conversations={[]}
        onClose={vi.fn<() => void>()}
        onDelete={vi.fn<(id: string) => void>()}
        onNewChat={vi.fn<() => void>()}
        onRename={vi.fn<(id: string, title: string) => void>()}
        onSelect={vi.fn<(id: string) => void>()}
        open
      />,
    );

    const link = screen.getByRole('link', {
      name: t('committee.sidebarLabel'),
    });

    expect(link).toBeVisible();
    expect(link).toHaveAttribute('href', '/committee');
    expect(link).toHaveClass('min-h-11');
  });

  it('closes the mobile drawer when the committee link is clicked', () => {
    const onClose = vi.fn<() => void>();
    render(
      <Sidebar
        activeId={null}
        conversations={[]}
        mobile
        onClose={onClose}
        onDelete={vi.fn<(id: string) => void>()}
        onNewChat={vi.fn<() => void>()}
        onRename={vi.fn<(id: string, title: string) => void>()}
        onSelect={vi.fn<(id: string) => void>()}
        open
      />,
    );

    fireEvent.click(
      screen.getByRole('link', { name: t('committee.sidebarLabel') }),
    );

    expect(closeSidebarOnMobileMock).toHaveBeenCalledExactlyOnceWith(onClose);
  });
});

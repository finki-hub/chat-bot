import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { Sidebar } from '@/components/shell/sidebar';

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
  it('uses modal dialog semantics for the mobile drawer', () => {
    render(
      <Sidebar
        {...baseProps}
        mobile
      />,
    );

    expect(
      screen.getByRole('dialog', { name: 'Странична лента' }),
    ).toBeInTheDocument();
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

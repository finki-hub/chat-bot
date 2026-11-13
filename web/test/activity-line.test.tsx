import { act, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ActivityLine } from '@/components/chat/activity-line';

describe('ActivityLine', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders the label as a live status', () => {
    render(<ActivityLine label="Обработка на прашањето" />);

    expect(screen.getByRole('status')).toHaveTextContent(
      'Обработка на прашањето…',
    );
    expect(screen.queryByTestId('elapsed')).not.toBeInTheDocument();
  });

  it('counts elapsed time and reassures once the wait drags on', () => {
    vi.useFakeTimers();
    render(<ActivityLine label="Обработка на прашањето" />);

    act(() => {
      vi.advanceTimersByTime(4_000);
    });

    expect(screen.getByTestId('elapsed')).toHaveTextContent('4 с');
    expect(screen.queryByTestId('activity-hint')).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(12_000);
    });

    expect(screen.getByTestId('activity-hint')).toHaveTextContent(
      'може да потрае',
    );

    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    expect(screen.getByTestId('elapsed')).toHaveTextContent('1 мин 16 с');
    expect(screen.getByTestId('activity-hint')).toHaveTextContent(
      'подолго од вообичаено',
    );
  });
});

describe('ActivityLine handover', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('keeps counting when the same wait moves to another component', () => {
    vi.useFakeTimers();
    const first = render(<ActivityLine label="Предавање" />);

    act(() => {
      vi.advanceTimersByTime(20_000);
    });
    first.unmount();
    render(<ActivityLine label="Предавање" />);

    expect(screen.getByTestId('elapsed')).toHaveTextContent('20 с');
    expect(screen.getByTestId('activity-hint')).toBeInTheDocument();
  });

  it('restarts the clock for a different phase', () => {
    vi.useFakeTimers();
    const first = render(<ActivityLine label="Прва фаза" />);

    act(() => {
      vi.advanceTimersByTime(20_000);
    });
    first.unmount();
    render(<ActivityLine label="Втора фаза" />);

    expect(screen.queryByTestId('elapsed')).not.toBeInTheDocument();
  });
});

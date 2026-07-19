import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ConfirmDialog } from '@/components/ui/confirm-dialog';

describe('ConfirmDialog', () => {
  it('clears a failed confirmation after a controlled close', async () => {
    const onConfirm = vi.fn<() => Promise<boolean>>().mockResolvedValue(false);
    const onOpenChange = vi.fn<(open: boolean) => void>();
    const dialog = (open: boolean) => (
      <ConfirmDialog
        confirmLabel="Избриши"
        description="Ова дејство не може да се врати."
        errorMessage="Бришењето не успеа."
        onConfirm={onConfirm}
        onOpenChange={onOpenChange}
        open={open}
        title="Потврди бришење"
      />
    );
    const { rerender } = render(dialog(true));

    fireEvent.click(screen.getByTestId('confirm-action'));

    await expect(screen.findByRole('alert')).resolves.toHaveTextContent(
      'Бришењето не успеа.',
    );

    rerender(dialog(false));
    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
    rerender(dialog(true));

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('ignores a late failure after a controlled close', async () => {
    let resolveConfirmation: ((confirmed: boolean) => void) | undefined;
    const onConfirm = vi.fn<() => Promise<boolean>>(
      () =>
        new Promise<boolean>((resolve) => {
          resolveConfirmation = resolve;
        }),
    );
    const dialog = (open: boolean) => (
      <ConfirmDialog
        confirmLabel="Избриши"
        description="Ова дејство не може да се врати."
        errorMessage="Бришењето не успеа."
        onConfirm={onConfirm}
        onOpenChange={vi.fn<(nextOpen: boolean) => void>()}
        open={open}
        title="Потврди бришење"
      />
    );
    const { rerender } = render(dialog(true));

    fireEvent.click(screen.getByTestId('confirm-action'));
    rerender(dialog(false));

    const finishConfirmation = resolveConfirmation;
    if (finishConfirmation === undefined) {
      throw new TypeError('confirmation did not start');
    }
    await act(async () => {
      finishConfirmation(false);
      await Promise.resolve();
    });
    rerender(dialog(true));

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

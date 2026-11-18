import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import {
  type FeedbackDetails,
  FeedbackReasonDialog,
} from '@/components/chat/feedback-reason-dialog';

const FAILED_SAVE = 'Не успеавме да го зачуваме одговорот. Обиди се повторно.';

describe('FeedbackReasonDialog', () => {
  it('clears a failed save when the dialog closes and reopens', async () => {
    const onOpenChange = vi.fn<(open: boolean) => void>();
    const onSubmit = vi
      .fn<(details: FeedbackDetails) => Promise<boolean>>()
      .mockResolvedValue(false);
    const { rerender } = render(
      <FeedbackReasonDialog
        onOpenChange={onOpenChange}
        onSubmit={onSubmit}
        open
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Застарено' }));
    fireEvent.click(screen.getByRole('button', { name: 'Испрати' }));

    const alert = await screen.findByRole('alert');

    expect(alert).toHaveTextContent(FAILED_SAVE);

    fireEvent.click(screen.getByRole('button', { name: 'Прескокни' }));
    rerender(
      <FeedbackReasonDialog
        onOpenChange={onOpenChange}
        onSubmit={onSubmit}
        open={false}
      />,
    );
    rerender(
      <FeedbackReasonDialog
        onOpenChange={onOpenChange}
        onSubmit={onSubmit}
        open
      />,
    );

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

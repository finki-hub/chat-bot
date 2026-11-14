import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { UserMessage } from '@/components/chat/user-message';

const QUESTION = 'Колку изнесува школарината?';
const EDIT = 'Измени прашање';

const question: MyUIMessage = {
  id: 'u1',
  metadata: {},
  parts: [{ text: QUESTION, type: 'text' }],
  role: 'user',
};

describe('UserMessage editing', () => {
  it('resends the edited text for the same message', () => {
    const onEdit = vi.fn<(id: string, text: string) => void>();
    render(
      <UserMessage
        message={question}
        onEdit={onEdit}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: EDIT }));

    const input = screen.getByRole('textbox', { name: 'Изменето прашање' });
    const resend = screen.getByRole('button', { name: 'Испрати повторно' });

    expect(input).toHaveValue(QUESTION);
    expect(input).toHaveFocus();
    // Unchanged text has nothing to resend.
    expect(resend).toBeDisabled();

    fireEvent.change(input, {
      target: { value: '  Колку изнесува школарината за вонредни?  ' },
    });
    fireEvent.click(resend);

    expect(onEdit).toHaveBeenCalledWith(
      'u1',
      'Колку изнесува школарината за вонредни?',
    );
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  });

  it('cancels with Escape without resending', () => {
    const onEdit = vi.fn<(id: string, text: string) => void>();
    render(
      <UserMessage
        message={question}
        onEdit={onEdit}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: EDIT }));
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' });

    expect(onEdit).not.toHaveBeenCalled();
    expect(screen.getByText(QUESTION)).toBeInTheDocument();
  });

  it('offers no edit while an answer is streaming', () => {
    const onEditMessage = vi.fn<(id: string, text: string) => void>();
    const view = render(
      <Thread
        messages={[question]}
        onEditMessage={onEditMessage}
        status="ready"
      />,
    );

    expect(screen.getByRole('button', { name: EDIT })).toBeInTheDocument();

    view.rerender(
      <Thread
        messages={[question]}
        onEditMessage={onEditMessage}
        status="streaming"
      />,
    );

    expect(
      screen.queryByRole('button', { name: EDIT }),
    ).not.toBeInTheDocument();
  });
});

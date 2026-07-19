import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/conversation-types';

import { ConversationList } from '@/components/shell/conversation-list';

const conversations: ConversationRow[] = [
  {
    id: 'c1',
    model: 'claude-sonnet-5',
    title: 'Прв разговор',
  },
];

const actionLabels = ['Генерирај име', 'Преименувај', 'Избриши'] as const;

describe('ConversationList action tooltips', () => {
  it.each(actionLabels)(
    'shows “%s” when its action is hovered',
    async (label) => {
      const user = userEvent.setup();
      const onDelete = vi.fn<(id: string) => void>();
      const onGenerateTitle = vi.fn<(id: string) => void>();
      const onRename = vi.fn<(id: string, title: string) => void>();
      const onSelect = vi.fn<(id: string) => void>();

      render(
        <ConversationList
          activeId={null}
          conversations={conversations}
          onDelete={onDelete}
          onGenerateTitle={onGenerateTitle}
          onRename={onRename}
          onSelect={onSelect}
        />,
      );

      const item = screen.getByTestId('conversation-c1');
      await user.hover(within(item).getByRole('button', { name: label }));

      await expect(screen.findByRole('tooltip')).resolves.toHaveTextContent(
        label,
      );
    },
  );
});

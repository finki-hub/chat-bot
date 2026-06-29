import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/db';

import { ConversationList } from '@/components/shell/conversation-list';

const conversations: ConversationRow[] = [
  {
    createdAt: 1,
    id: 'c1',
    model: 'claude-sonnet-4-6',
    title: 'Прв разговор',
    updatedAt: 2,
  },
];

const noop = vi.fn<() => void>;

describe('ConversationList accessibility', () => {
  it('keeps row actions discoverable when a keyboard user focuses the row', () => {
    render(
      <ConversationList
        activeId={null}
        conversations={conversations}
        onDelete={noop()}
        onRename={noop()}
        onSelect={noop()}
      />,
    );

    const item = screen.getByTestId('conversation-c1');
    const actions = within(item).getByTestId('row-actions');

    expect(actions).toHaveClass('sm:group-focus-within:opacity-100');
  });
});

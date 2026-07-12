import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/conversation-types';

import { ConversationList } from '@/components/shell/conversation-list';

const conversations: ConversationRow[] = [
  {
    id: 'c1',
    model: 'claude-sonnet-5',
    title: 'Прв разговор',
  },
  {
    id: 'c2',
    model: 'gpt-5.4-mini',
    title: 'Втор разговор',
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

  it('gives mobile conversation controls 44px touch targets', () => {
    render(
      <ConversationList
        activeId={null}
        conversations={conversations}
        onDelete={noop()}
        onGenerateTitle={noop()}
        onRename={noop()}
        onSelect={noop()}
      />,
    );

    const item = screen.getByTestId('conversation-c1');

    expect(
      within(item).getByRole('button', { name: 'Прв разговор' }),
    ).toHaveClass('min-h-11');
    expect(
      within(item).getByRole('button', { name: 'Генерирај име' }),
    ).toHaveClass('size-11');
    expect(
      within(item).getByRole('button', { name: 'Преименувај' }),
    ).toHaveClass('size-11');
    expect(within(item).getByRole('button', { name: 'Избриши' })).toHaveClass(
      'size-11',
    );
  });

  it('disables every title generation action while one title is generating', () => {
    render(
      <ConversationList
        activeId={null}
        conversations={conversations}
        generatingTitleId="c1"
        onDelete={noop()}
        onGenerateTitle={noop()}
        onRename={noop()}
        onSelect={noop()}
      />,
    );

    const buttons = screen.getAllByRole('button', { name: 'Генерирај име' });

    expect(buttons).toHaveLength(2);

    for (const button of buttons) {
      expect(button).toBeDisabled();
    }
  });
});

import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/conversation-types';

import { ConversationList } from '@/components/shell/conversation-list';

const conversations: ConversationRow[] = [
  {
    id: 'c1',
    model: 'claude-sonnet-5-5',
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
  it('uses the whole row as the conversation selection target', () => {
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
    const selection = within(item).getByRole('button', {
      name: 'Прв разговор',
    });

    expect(item).toHaveClass('relative');
    expect(selection).toHaveClass('before:absolute', 'before:inset-0');
    expect(within(item).getByTestId('row-actions').parentElement).toHaveClass(
      'relative',
      'z-10',
    );
  });

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

    expect(actions).toHaveClass(
      'pointer-fine:opacity-0',
      'pointer-fine:group-focus-within:opacity-100',
    );
  });

  it('uses one menu trigger per row, 44px on touch, with labeled actions', async () => {
    const user = userEvent.setup();
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
    const actionsTrigger = within(item).getByRole('button', {
      name: 'Дејства за разговорот: Прв разговор',
    });

    expect(
      within(item).getByRole('button', { name: 'Прв разговор' }),
    ).toHaveClass('min-h-11', 'pointer-fine:min-h-0');
    expect(actionsTrigger).toHaveClass('size-12', 'pointer-fine:size-7');
    // The trigger is the row's only control besides the title.
    expect(within(item).getAllByRole('button')).toHaveLength(2);

    await user.click(actionsTrigger);

    const menu = screen.getByRole('menu');

    expect(
      within(menu).getByRole('menuitem', { name: 'Генерирај име' }),
    ).toHaveClass('min-h-11', 'pointer-fine:min-h-8');
    expect(
      within(menu).getByRole('menuitem', { name: 'Преименувај' }),
    ).toHaveClass('min-h-11', 'pointer-fine:min-h-8');
    expect(within(menu).getByRole('menuitem', { name: 'Избриши' })).toHaveClass(
      'min-h-11',
      'pointer-fine:min-h-8',
    );
  });

  it('disables title generation elsewhere while one title is generating', () => {
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

    const generating = within(
      screen.getByTestId('conversation-c1'),
    ).getByTestId('row-actions');

    expect(generating).toHaveAttribute('aria-busy', 'true');

    fireEvent.keyDown(
      within(screen.getByTestId('conversation-c2')).getByTestId('row-actions'),
      { key: 'Enter' },
    );

    expect(
      screen.getByRole('menuitem', { name: 'Генерирај име' }),
    ).toHaveAttribute('aria-disabled', 'true');
  });
});

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
      'hidden',
      'pointer-fine:flex',
      'pointer-fine:group-focus-within:opacity-100',
    );
  });

  it('uses one 44px mobile menu trigger with labeled actions', async () => {
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
    expect(actionsTrigger).toHaveClass('size-12', 'pointer-fine:hidden');

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

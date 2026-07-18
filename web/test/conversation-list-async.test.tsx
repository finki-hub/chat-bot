import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/conversation-types';

import { ConversationList } from '@/components/shell/conversation-list';
import { Sidebar } from '@/components/shell/sidebar';

const conversation: ConversationRow = {
  id: 'conversation-1',
  model: 'gpt-5.4-mini',
  title: 'Прв разговор',
};

const CONFIRM_ACTION_TEST_ID = 'confirm-action';
const CONVERSATION_ROW_TEST_ID = 'conversation-conversation-1';
const DELETE_DIALOG_TITLE = 'Избриши разговор?';
const noop = vi.fn<() => void>;

type Deferred = {
  readonly promise: Promise<void>;
  readonly resolve: () => void;
};

const createDeferred = (): Deferred => {
  let resolvePromise: (() => void) | undefined;
  const promise = new Promise<void>((resolve) => {
    resolvePromise = resolve;
  });
  if (resolvePromise === undefined) {
    throw new Error('Deferred resolver was not captured');
  }
  return { promise, resolve: resolvePromise };
};

describe('ConversationList async actions', () => {
  it('prevents duplicate deletion while confirmation is pending', async () => {
    const deferred = createDeferred();
    const onDelete = vi.fn<(id: string) => Promise<void>>(
      () => deferred.promise,
    );
    render(
      <ConversationList
        activeId={null}
        conversations={[conversation]}
        onDelete={onDelete}
        onRename={noop()}
        onSelect={noop()}
      />,
    );
    const row = screen.getByTestId(CONVERSATION_ROW_TEST_ID);

    fireEvent.click(within(row).getByRole('button', { name: 'Избриши' }));
    const dialog = await screen.findByRole('dialog', {
      name: DELETE_DIALOG_TITLE,
    });
    const confirm = within(dialog).getByTestId(CONFIRM_ACTION_TEST_ID);
    fireEvent.click(confirm);

    await waitFor(() => {
      expect(onDelete).toHaveBeenCalledOnce();
    });

    expect(confirm).toBeDisabled();

    expect(confirm).toHaveAttribute('aria-busy', 'true');

    fireEvent.click(confirm);

    expect(onDelete).toHaveBeenCalledOnce();

    await act(async () => {
      deferred.resolve();
      await deferred.promise;
    });
    await waitFor(() => {
      expect(
        screen.queryByRole('dialog', { name: DELETE_DIALOG_TITLE }),
      ).not.toBeInTheDocument();
    });
  });

  it('retains the rename input while saving is pending', async () => {
    const deferred = createDeferred();
    const onRename = vi.fn<(id: string, title: string) => Promise<void>>(
      () => deferred.promise,
    );
    render(
      <ConversationList
        activeId={null}
        conversations={[conversation]}
        onDelete={noop()}
        onRename={onRename}
        onSelect={noop()}
      />,
    );
    const row = screen.getByTestId(CONVERSATION_ROW_TEST_ID);
    fireEvent.click(within(row).getByRole('button', { name: 'Преименувај' }));
    const input = screen.getByLabelText('Ново име на разговорот');
    fireEvent.change(input, { target: { value: 'Ново име' } });

    fireEvent.click(screen.getByTestId('confirm-rename'));

    await waitFor(() => {
      expect(onRename).toHaveBeenCalledWith('conversation-1', 'Ново име');
    });

    expect(input).toBeDisabled();

    expect(input).toHaveValue('Ново име');

    expect(screen.getByTestId('confirm-rename')).toHaveAttribute(
      'aria-busy',
      'true',
    );

    await act(async () => {
      deferred.resolve();
      await deferred.promise;
    });
    await waitFor(() => {
      expect(
        screen.queryByRole('dialog', { name: 'Преименувај разговор' }),
      ).not.toBeInTheDocument();
    });
  });

  it('keeps deletion confirmation open when deletion fails', async () => {
    const onDelete = vi.fn<(id: string) => Promise<boolean>>(() =>
      Promise.resolve(false),
    );
    render(
      <ConversationList
        activeId={null}
        conversations={[conversation]}
        onDelete={onDelete}
        onRename={noop()}
        onSelect={noop()}
      />,
    );
    const row = screen.getByTestId(CONVERSATION_ROW_TEST_ID);
    fireEvent.click(within(row).getByRole('button', { name: 'Избриши' }));
    const dialog = await screen.findByRole('dialog', {
      name: DELETE_DIALOG_TITLE,
    });

    fireEvent.click(within(dialog).getByTestId(CONFIRM_ACTION_TEST_ID));

    const alert = await within(dialog).findByText(
      'Разговорот не можеше да се избрише.',
    );

    expect(alert).toHaveAttribute('role', 'alert');

    expect(dialog).toBeInTheDocument();

    expect(within(dialog).getByTestId(CONFIRM_ACTION_TEST_ID)).toBeEnabled();
  });

  it('retains the rename input when saving fails', async () => {
    const onRename = vi.fn<(id: string, title: string) => Promise<boolean>>(
      () => Promise.resolve(false),
    );
    render(
      <ConversationList
        activeId={null}
        conversations={[conversation]}
        onDelete={noop()}
        onRename={onRename}
        onSelect={noop()}
      />,
    );
    const row = screen.getByTestId(CONVERSATION_ROW_TEST_ID);
    fireEvent.click(within(row).getByRole('button', { name: 'Преименувај' }));
    const input = screen.getByLabelText('Ново име на разговорот');
    fireEvent.change(input, { target: { value: 'Ново име' } });

    fireEvent.click(screen.getByTestId('confirm-rename'));

    const alert = await screen.findByText(
      'Разговорот не можеше да се преименува.',
    );

    expect(alert).toHaveAttribute('role', 'alert');

    expect(input).toHaveValue('Ново име');

    expect(input).toBeEnabled();

    expect(
      screen.getByRole('dialog', { name: 'Преименувај разговор' }),
    ).toBeInTheDocument();
  });
});

describe('Sidebar async actions', () => {
  it('keeps delete-all confirmation open when clearing fails', async () => {
    const onClearAll = vi.fn<() => Promise<boolean>>(() =>
      Promise.resolve(false),
    );
    render(
      <Sidebar
        activeId={null}
        conversations={[conversation]}
        onClearAll={onClearAll}
        onClose={noop()}
        onDelete={noop()}
        onNewChat={noop()}
        onRename={noop()}
        onSelect={noop()}
        open
      />,
    );
    fireEvent.click(screen.getByTestId('delete-all'));
    const dialog = await screen.findByRole('dialog', {
      name: 'Избриши ги сите разговори?',
    });

    fireEvent.click(within(dialog).getByTestId(CONFIRM_ACTION_TEST_ID));

    const alert = await within(dialog).findByText(
      'Разговорите не можеа да се избришат.',
    );

    expect(alert).toHaveAttribute('role', 'alert');

    expect(dialog).toBeInTheDocument();

    expect(within(dialog).getByTestId(CONFIRM_ACTION_TEST_ID)).toBeEnabled();
  });
});

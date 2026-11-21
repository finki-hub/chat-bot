import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ConversationRow } from '@/lib/conversation-types';

import { Sidebar } from '@/components/shell/sidebar';

const ROWS: ConversationRow[] = [
  { id: 'c1', model: 'gpt-6-luna', title: 'Испити' },
  { id: 'c2', model: 'gpt-6-luna', title: 'Упис' },
];

const FIRST_TITLE_PATTERN = /Испити/u;
const SECOND_TITLE_PATTERN = /Упис/u;
const RETRY_PATTERN = /Обиди се повторно/u;

const searchBody = {
  query: 'испити',
  results: [
    {
      conversationId: 'c1',
      conversationTitle: 'Испити',
      conversationUpdatedAt: '2026-11-02T12:00:00Z',
      message: {
        createdAt: '2026-11-02T12:00:00Z',
        id: 'm1',
        role: 'assistant',
      },
      snippet: 'Рокот за \u{1}испити\u{2} е објавен.',
    },
    {
      conversationId: 'c2',
      conversationTitle: 'Упис',
      conversationUpdatedAt: '2026-11-01T12:00:00Z',
      message: null,
      snippet: null,
    },
  ],
};

const renderSidebar = (
  onSelectMessage = vi.fn<
    (conversationId: string, messageId: null | string) => void
  >(),
) =>
  render(
    <Sidebar
      activeId={null}
      conversations={ROWS}
      onClose={vi.fn<() => void>()}
      onDelete={vi.fn<(...args: string[]) => void>()}
      onNewChat={vi.fn<() => void>()}
      onRename={vi.fn<(...args: string[]) => void>()}
      onSelect={vi.fn<(...args: string[]) => void>()}
      onSelectMessage={onSelectMessage}
      open
    />,
  );

describe('Sidebar conversation search', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>().mockResolvedValue(Response.json(searchBody)),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders message snippets, highlights matches, and opens the message', async () => {
    const onSelectMessage =
      vi.fn<(conversationId: string, messageId: null | string) => void>();
    renderSidebar(onSelectMessage);

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'испити' },
    });

    const results = await screen.findByTestId('conversation-search-results');
    const firstResult = within(results).getByRole('button', {
      name: FIRST_TITLE_PATTERN,
    });

    expect(within(firstResult).getByText('испити')).toBeInTheDocument();

    fireEvent.click(firstResult);

    expect(onSelectMessage).toHaveBeenCalledWith('c1', 'm1');
  });

  it('renders title-only matches without a message jump', async () => {
    const onSelectMessage =
      vi.fn<(conversationId: string, messageId: null | string) => void>();
    renderSidebar(onSelectMessage);

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'испити' },
    });

    const results = await screen.findByTestId('conversation-search-results');
    const titleOnly = within(results).getByRole('button', {
      name: SECOND_TITLE_PATTERN,
    });

    expect(within(titleOnly).getByText('Наслов')).toBeInTheDocument();

    fireEvent.click(titleOnly);

    expect(onSelectMessage).toHaveBeenCalledWith('c2', null);
  });

  it('shows the search error and can retry', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError('offline'));
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'испити' },
    });

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Пребарувањето не успеа',
      );
    });

    fireEvent.click(screen.getByRole('button', { name: RETRY_PATTERN }));

    await expect(
      screen.findByTestId('conversation-search-results'),
    ).resolves.toBeInTheDocument();
  });
});

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
const QUERY = 'испити';
const RESULTS_TEST_ID = 'conversation-search-results';

const searchBody = {
  query: QUERY,
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
    (
      conversationId: string,
      messageId: null | string,
      conversationTitle: null | string,
    ) => void
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
      vi.fn<
        (
          conversationId: string,
          messageId: null | string,
          conversationTitle: null | string,
        ) => void
      >();
    renderSidebar(onSelectMessage);

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: QUERY },
    });

    const results = await screen.findByTestId(RESULTS_TEST_ID);
    const firstResult = within(results).getByRole('button', {
      name: FIRST_TITLE_PATTERN,
    });

    expect(within(firstResult).getByText('испити')).toBeInTheDocument();

    fireEvent.click(firstResult);

    expect(onSelectMessage).toHaveBeenCalledWith('c1', 'm1', 'Испити');
  });

  it('renders title-only matches without a message jump', async () => {
    const onSelectMessage =
      vi.fn<
        (
          conversationId: string,
          messageId: null | string,
          conversationTitle: null | string,
        ) => void
      >();
    renderSidebar(onSelectMessage);

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: QUERY },
    });

    const results = await screen.findByTestId(RESULTS_TEST_ID);
    const titleOnly = within(results).getByRole('button', {
      name: SECOND_TITLE_PATTERN,
    });

    expect(within(titleOnly).getByText('Наслов')).toBeInTheDocument();

    fireEvent.click(titleOnly);

    expect(onSelectMessage).toHaveBeenCalledWith('c2', null, 'Упис');
  });

  it('keeps local title matches when the server has no results', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      Response.json({ query: 'исп', results: [] }),
    );
    const onSelectMessage =
      vi.fn<
        (
          conversationId: string,
          messageId: null | string,
          conversationTitle: null | string,
        ) => void
      >();
    renderSidebar(onSelectMessage);

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'исп' },
    });

    const results = await screen.findByTestId(RESULTS_TEST_ID);
    const localMatch = within(results).getByRole('button', {
      name: FIRST_TITLE_PATTERN,
    });

    expect(within(localMatch).getByText('Наслов')).toBeInTheDocument();

    fireEvent.click(localMatch);

    expect(onSelectMessage).toHaveBeenCalledWith('c1', null, 'Испити');
  });

  it('shows the search error and can retry', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError('offline'));
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: QUERY },
    });

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Пребарувањето не успеа',
      );
    });

    fireEvent.click(screen.getByRole('button', { name: RETRY_PATTERN }));

    await expect(
      screen.findByTestId(RESULTS_TEST_ID),
    ).resolves.toBeInTheDocument();
  });

  it('shows who wrote the matched message', async () => {
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: QUERY },
    });

    const results = await screen.findByTestId(RESULTS_TEST_ID);
    const firstResult = within(results).getByRole('button', {
      name: FIRST_TITLE_PATTERN,
    });

    expect(firstResult).toHaveTextContent('Асистент');
  });

  it('caps the query at the API limit', () => {
    renderSidebar();

    expect(screen.getByRole('searchbox')).toHaveAttribute('maxlength', '200');
  });

  it('loads the next page of results', async () => {
    const page = Array.from({ length: 20 }, (_, index) => ({
      conversationId: `p${index}`,
      conversationTitle: `Резултат ${index}`,
      conversationUpdatedAt: '2026-11-02T12:00:00Z',
      message: null,
      snippet: null,
    }));
    vi.mocked(fetch)
      .mockResolvedValueOnce(Response.json({ query: QUERY, results: page }))
      .mockResolvedValueOnce(
        Response.json({
          query: QUERY,
          results: [
            {
              ...page[0],
              conversationId: 'last',
              conversationTitle: 'Последен',
            },
          ],
        }),
      );
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: QUERY },
    });
    fireEvent.click(
      await screen.findByRole('button', { name: 'Покажи повеќе' }),
    );

    await expect(screen.findByText('Последен')).resolves.toBeInTheDocument();

    const secondUrl = vi.mocked(fetch).mock.calls[1]?.[0];

    expect(typeof secondUrl === 'string' ? secondUrl : '').toContain(
      'offset=20',
    );
    expect(
      screen.queryByRole('button', { name: 'Покажи повеќе' }),
    ).not.toBeInTheDocument();
  });

  it('opens the sidebar and focuses search on Ctrl+K', async () => {
    const onOpen = vi.fn<() => void>();
    const props = {
      activeId: null,
      conversations: ROWS,
      onClose: vi.fn<() => void>(),
      onDelete: vi.fn<(...args: string[]) => void>(),
      onNewChat: vi.fn<() => void>(),
      onOpen,
      onRename: vi.fn<(...args: string[]) => void>(),
      onSelect: vi.fn<(...args: string[]) => void>(),
    };
    const { rerender } = render(
      <Sidebar
        {...props}
        open={false}
      />,
    );

    fireEvent.keyDown(document, { ctrlKey: true, key: 'k' });

    expect(onOpen).toHaveBeenCalledOnce();

    rerender(
      <Sidebar
        {...props}
        onOpen={vi.fn<() => void>()}
        open
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole('searchbox')).toHaveFocus();
    });
  });
});

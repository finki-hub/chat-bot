import type { ReactNode } from 'react';

import type { ChatSearchResult } from '@/lib/conversation-types';

import { t } from '@/lib/i18n';

const HIGHLIGHT_START = '\u{1}';
const HIGHLIGHT_END = '\u{2}';

const renderSnippet = (snippet: string): ReactNode[] => {
  const nodes: ReactNode[] = [];
  let index = 0;
  let key = 0;

  while (index < snippet.length) {
    const start = snippet.indexOf(HIGHLIGHT_START, index);
    if (start === -1) {
      nodes.push(<span key={key}>{snippet.slice(index)}</span>);
      break;
    }
    if (start > index) {
      nodes.push(<span key={key}>{snippet.slice(index, start)}</span>);
      key += 1;
    }
    const end = snippet.indexOf(HIGHLIGHT_END, start + 1);
    if (end === -1) {
      nodes.push(<span key={key}>{snippet.slice(start + 1)}</span>);
      break;
    }
    nodes.push(
      <mark
        className="rounded-sm bg-primary/15 px-0.5 text-foreground"
        key={key}
      >
        {snippet.slice(start + 1, end)}
      </mark>,
    );
    key += 1;
    index = end + 1;
  }

  return nodes;
};

export const ConversationSearchResults = ({
  onSelect,
  results,
}: {
  onSelect: (
    conversationId: string,
    messageId: null | string,
    conversationTitle: null | string,
  ) => void;
  results: readonly ChatSearchResult[];
}) => (
  <ul
    aria-label={t('search.results')}
    className="flex flex-col gap-1"
    data-testid="conversation-search-results"
  >
    {results.map((result) => (
      <li key={`${result.conversationId}:${result.message?.id ?? 'title'}`}>
        <button
          className="flex min-h-11 w-full flex-col gap-1 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => {
            onSelect(
              result.conversationId,
              result.message?.id ?? null,
              result.conversationTitle,
            );
          }}
          type="button"
        >
          <span className="flex min-w-0 items-center gap-2">
            <span className="truncate text-sm font-medium text-foreground">
              {result.conversationTitle ?? t('conversation.untitled')}
            </span>
            {result.message === null ? (
              <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                {t('search.titleMatch')}
              </span>
            ) : null}
          </span>
          {result.snippet ? (
            <span className="line-clamp-2 text-xs leading-snug text-muted-foreground">
              {renderSnippet(result.snippet)}
            </span>
          ) : null}
        </button>
      </li>
    ))}
  </ul>
);

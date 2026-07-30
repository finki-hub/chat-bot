'use client';

import { BookOpenText, ChevronRight, ExternalLink } from 'lucide-react';
import { useId, useState } from 'react';

import type { RetrievedSource } from '@/lib/api-types';

import {
  MessageResponse,
  type MessageResponseProps,
} from '@/components/ai-elements/message';
import { t } from '@/lib/i18n';

const AUTO_EXPAND_SOURCE_LIMIT = 2;
type MarkdownComponents = NonNullable<MessageResponseProps['components']>;

const sourceMarkdownComponents = {
  img: () => null,
  strong: ({ children }) => <strong>{children}</strong>,
} satisfies MarkdownComponents;

const collapsedSourceMarkdownComponents = {
  ...sourceMarkdownComponents,
  a: ({ children }) => (
    <span className="wrap-anywhere font-medium text-primary underline">
      {children}
    </span>
  ),
} satisfies MarkdownComponents;

const sourceMarkdownPlugins = {} satisfies NonNullable<
  MessageResponseProps['plugins']
>;
const disallowedSourceMarkdownElements = [
  'img',
  'picture',
  'source',
] as const satisfies NonNullable<MessageResponseProps['disallowedElements']>;

const SourceKindLabel = ({ source }: { source: RetrievedSource }) => (
  <span className="rounded-full border border-border/70 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
    {source.kind === 'faq' ? t('sources.faq') : t('sources.chunk')}
  </span>
);

const SourceCard = ({ source }: { source: RetrievedSource }) => {
  const [expanded, setExpanded] = useState(false);
  const snippet = source.snippet ?? '';
  const hasSnippet = snippet.length > 0;
  const links = source.links ?? [];
  const snippetId = useId();
  const title = source.section
    ? `${source.title} · ${source.section}`
    : source.title;
  const content = (
    <>
      <span className="flex flex-wrap items-center gap-2">
        <SourceKindLabel source={source} />
        {typeof source.chunkIndex === 'number' ? (
          <span className="text-[10px] text-muted-foreground/70">
            #{source.chunkIndex + 1}
          </span>
        ) : null}
        {hasSnippet ? (
          <ChevronRight
            aria-hidden="true"
            className={`size-3 text-muted-foreground/70 transition-transform ${expanded ? 'rotate-90' : ''}`}
          />
        ) : null}
      </span>
      <span className="block line-clamp-2 text-sm font-medium leading-snug text-foreground">
        {title}
      </span>
    </>
  );

  return (
    <li className="min-w-0 rounded-lg border border-border/70 bg-muted/20 p-3 transition-colors hover:bg-muted/35">
      <div className="flex items-start justify-between gap-3">
        {hasSnippet ? (
          <div className="min-w-0 flex-1">
            <button
              aria-controls={snippetId}
              aria-expanded={expanded}
              className="min-h-11 w-full rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-0"
              onClick={() => {
                setExpanded((current) => !current);
              }}
              type="button"
            >
              {content}
            </button>
            <div
              aria-hidden={expanded ? undefined : true}
              className={`mt-2 text-xs leading-relaxed text-muted-foreground ${expanded ? 'whitespace-pre-wrap' : 'line-clamp-2'}`}
              id={snippetId}
            >
              <MessageResponse
                className="h-auto"
                components={
                  expanded
                    ? sourceMarkdownComponents
                    : collapsedSourceMarkdownComponents
                }
                controls={false}
                disallowedElements={disallowedSourceMarkdownElements}
                key={expanded ? 'expanded' : 'collapsed'}
                mode="static"
                plugins={sourceMarkdownPlugins}
              >
                {snippet}
              </MessageResponse>
            </div>
          </div>
        ) : (
          <div className="min-w-0 flex-1 space-y-1">{content}</div>
        )}
        {links.length > 0 ? (
          <div className="flex shrink-0 flex-wrap justify-end gap-1">
            {links.map((link) => (
              <a
                aria-label={`${t('sources.link')}: ${link.label}`}
                className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-md text-muted-foreground hover:bg-background hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-0 pointer-fine:min-w-0 pointer-fine:p-1"
                href={link.url}
                key={`${link.label}:${link.url}`}
                rel="noreferrer"
                target="_blank"
              >
                <ExternalLink
                  aria-hidden="true"
                  className="size-3.5"
                />
              </a>
            ))}
          </div>
        ) : null}
      </div>
    </li>
  );
};

export const SourceCards = ({
  complete,
  sources,
}: {
  complete: boolean;
  sources: readonly RetrievedSource[];
}) => {
  const sourceCount = sources.length;
  const [disclosure, setDisclosure] = useState<'automatic' | 'closed' | 'open'>(
    'automatic',
  );
  const open =
    disclosure === 'open' ||
    (disclosure === 'automatic' &&
      complete &&
      sourceCount > 0 &&
      sourceCount <= AUTO_EXPAND_SOURCE_LIMIT);
  const panelId = useId();

  if (sourceCount === 0) {
    return null;
  }

  const toggleDisclosure = () => {
    setDisclosure(open ? 'closed' : 'open');
  };

  return (
    <section
      aria-label={t('sources.title')}
      className="mt-3 space-y-2"
      data-testid="message-sources"
    >
      <button
        aria-controls={panelId}
        aria-expanded={open}
        aria-label={open ? t('sources.hide') : t('sources.show')}
        className="inline-flex min-h-11 items-center gap-1.5 rounded-md px-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-0 pointer-fine:px-0 pointer-fine:text-xs"
        onClick={toggleDisclosure}
        type="button"
      >
        <ChevronRight
          aria-hidden="true"
          className={`size-3 transition-transform ${open ? 'rotate-90' : ''}`}
        />
        <BookOpenText
          aria-hidden="true"
          className="size-3.5"
        />
        <span>{t('sources.title')}</span>
        <span className="rounded-full border border-border/70 px-1.5 py-0.5 text-[10px] tabular-nums text-muted-foreground/70">
          {sourceCount}
        </span>
      </button>
      {open ? (
        <ul
          className="grid items-start gap-2 sm:grid-cols-2"
          id={panelId}
        >
          {sources.map((source) => (
            <SourceCard
              key={`${source.kind}:${source.id}`}
              source={source}
            />
          ))}
        </ul>
      ) : null}
    </section>
  );
};

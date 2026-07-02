'use client';

import { BookOpenText, ChevronRight, ExternalLink } from 'lucide-react';
import { useId, useState } from 'react';

import type { RetrievedSource } from '@/lib/api-types';

import { t } from '@/lib/i18n';

const PREVIEW_WORD_LIMIT = 12;
const SENTENCE_END_RE = /[.!?]/u;
const WHITESPACE_RE = /\s+/u;

const snippetPreview = (snippet: string): string => {
  const sentenceEnd = snippet.search(SENTENCE_END_RE);
  if (sentenceEnd >= 0) {
    return snippet.slice(0, sentenceEnd + 1);
  }

  const words = snippet.trim().split(WHITESPACE_RE);
  const preview = words.slice(0, PREVIEW_WORD_LIMIT).join(' ');
  return words.length > PREVIEW_WORD_LIMIT ? `${preview}…` : preview;
};

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
  const visibleSnippet = expanded ? snippet : snippetPreview(snippet);
  const title = source.section
    ? `${source.title} · ${source.section}`
    : source.title;
  const content = (
    <>
      <div className="flex flex-wrap items-center gap-2">
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
      </div>
      <p className="line-clamp-2 text-sm font-medium leading-snug text-foreground">
        {title}
      </p>
      {hasSnippet ? (
        <p
          className={`mt-2 text-xs leading-relaxed text-muted-foreground ${expanded ? 'whitespace-pre-wrap' : 'line-clamp-2'}`}
          id={snippetId}
        >
          {visibleSnippet}
        </p>
      ) : null}
    </>
  );

  return (
    <li className="min-w-0 rounded-lg border border-border/70 bg-muted/20 p-3 transition-colors hover:bg-muted/35">
      <div className="flex items-start justify-between gap-3">
        {hasSnippet ? (
          <button
            aria-controls={snippetId}
            aria-expanded={expanded}
            className="min-w-0 flex-1 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onClick={() => {
              setExpanded((current) => !current);
            }}
            type="button"
          >
            {content}
          </button>
        ) : (
          <div className="min-w-0 flex-1 space-y-1">{content}</div>
        )}
        {links.length > 0 ? (
          <div className="flex shrink-0 flex-wrap justify-end gap-1">
            {links.map((link) => (
              <a
                aria-label={`${t('sources.link')}: ${link.label}`}
                className="rounded-md p-1 text-muted-foreground hover:bg-background hover:text-foreground"
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
  sources,
}: {
  sources: readonly RetrievedSource[];
}) => {
  const [open, setOpen] = useState(false);
  const panelId = useId();

  if (sources.length === 0) {
    return null;
  }

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
        className="inline-flex items-center gap-1.5 rounded-md text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => {
          setOpen((current) => !current);
        }}
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
          {sources.length}
        </span>
      </button>
      {open ? (
        <ul
          className="grid gap-2 sm:grid-cols-2"
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

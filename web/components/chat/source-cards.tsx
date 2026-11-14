'use client';

import {
  BookOpenText,
  ChevronRight,
  ExternalLink,
  FileText,
  MessageCircleQuestion,
  TriangleAlert,
} from 'lucide-react';
import { useEffect, useId, useRef, useState } from 'react';

import type { RetrievedSource, RetrievedSourceLink } from '@/lib/api-types';

import {
  MessageResponse,
  type MessageResponseProps,
} from '@/components/ai-elements/message';
import { groupRef } from '@/lib/citations';
import { t } from '@/lib/i18n';
import {
  formatDocumentDate,
  groupLinks,
  groupSections,
  groupSources,
  isOutdatedSource,
  sourceLinkText,
} from '@/lib/source-groups';

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

const SourceKindIcon = ({
  className,
  source,
}: {
  className: string;
  source: RetrievedSource;
}) => {
  const Icon = source.kind === 'faq' ? MessageCircleQuestion : FileText;

  return (
    <Icon
      aria-hidden="true"
      className={className}
    />
  );
};

// Kind, document date and a warning when the corpus marks the document stale.
const SourceMeta = ({
  citeRef,
  source,
}: {
  citeRef: number | undefined;
  source: RetrievedSource;
}) => {
  const date = formatDocumentDate(source.documentDate);

  return (
    <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-muted-foreground">
      {citeRef === undefined ? null : (
        // Matches the `[N]` chips in the answer.
        <span className="inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-muted px-1 text-[10px] font-medium tabular-nums">
          {citeRef}
        </span>
      )}
      <SourceKindIcon
        className="size-3.5 shrink-0"
        source={source}
      />
      <span>
        {source.kind === 'faq' ? t('sources.faq') : t('sources.chunk')}
      </span>
      {date === null ? null : (
        <>
          <span aria-hidden="true">·</span>
          <time
            dateTime={source.documentDate}
            title={t('sources.date')}
          >
            {date}
          </time>
        </>
      )}
      {isOutdatedSource(source) ? (
        <span
          className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-400"
          data-testid="source-outdated"
          title={t('sources.outdatedHint')}
        >
          <TriangleAlert
            aria-hidden="true"
            className="size-3"
          />
          {t('sources.outdated')}
        </span>
      ) : null}
    </span>
  );
};

// The badge's tooltip is hover-only, so the reason is also spelled out.
const OutdatedNote = ({ source }: { source: RetrievedSource }) =>
  isOutdatedSource(source) ? (
    <p
      className="mt-1.5 text-xs leading-snug text-amber-700 dark:text-amber-400"
      data-testid="source-outdated-note"
    >
      {t('sources.outdatedHint')}
    </p>
  ) : null;

const SourceTitle = ({
  expandable,
  expanded,
  title,
}: {
  expandable: boolean;
  expanded: boolean;
  title: string;
}) => (
  <span className="mt-1 flex items-start justify-between gap-2">
    <span className="line-clamp-2 text-sm font-medium leading-snug text-foreground">
      {title}
    </span>
    {expandable ? (
      <ChevronRight
        aria-hidden="true"
        className={`mt-0.5 size-3.5 shrink-0 text-muted-foreground/70 transition-transform ${expanded ? 'rotate-90' : ''}`}
      />
    ) : null}
  </span>
);

const SourceLinks = ({
  links,
  title,
}: {
  links: readonly RetrievedSourceLink[];
  title: string;
}) =>
  links.length > 0 ? (
    <div className="mt-2 flex flex-wrap gap-x-3">
      {links.map((link) => (
        <a
          aria-label={`${t('sources.link')}: ${link.label}`}
          className="inline-flex min-h-11 items-center gap-1 rounded-sm text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-0"
          href={link.url}
          key={`${link.label}:${link.url}`}
          rel="noreferrer"
          target="_blank"
        >
          <ExternalLink
            aria-hidden="true"
            className="size-3.5 shrink-0"
          />
          <span className="max-w-48 truncate">
            {sourceLinkText(link, title)}
          </span>
        </a>
      ))}
    </div>
  ) : null;

const CARD_CLASS =
  'min-w-0 rounded-lg border border-border/70 bg-muted/20 p-3 transition-colors hover:bg-muted/35';
const CARD_TOGGLE_CLASS =
  'block min-h-11 w-full rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-0';

type CardCitation = {
  citeRef: number | undefined;
  focused: boolean;
};

const cardClass = (focused: boolean): string =>
  focused ? `${CARD_CLASS} ring-2 ring-primary/50` : CARD_CLASS;

const SourceCard = ({
  citeRef,
  focused,
  source,
}: CardCitation & { source: RetrievedSource }) => {
  const [expanded, setExpanded] = useState(false);
  const snippet = source.snippet ?? '';
  const hasSnippet = snippet.length > 0;
  const snippetId = useId();
  const title = source.section
    ? `${source.title} · ${source.section}`
    : source.title;
  const heading = (
    <>
      <SourceMeta
        citeRef={citeRef}
        source={source}
      />
      <SourceTitle
        expandable={hasSnippet}
        expanded={expanded}
        title={title}
      />
    </>
  );

  return (
    <li
      className={cardClass(focused)}
      data-ref={citeRef}
    >
      {hasSnippet ? (
        <>
          <button
            aria-controls={snippetId}
            aria-expanded={expanded}
            className={CARD_TOGGLE_CLASS}
            onClick={() => {
              setExpanded((current) => !current);
            }}
            type="button"
          >
            {heading}
          </button>
          <div
            aria-hidden={expanded ? undefined : true}
            className={`mt-1.5 text-xs leading-relaxed text-muted-foreground ${expanded ? 'whitespace-pre-wrap' : 'line-clamp-2'}`}
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
        </>
      ) : (
        <div>{heading}</div>
      )}
      <OutdatedNote source={source} />
      <SourceLinks
        links={source.links ?? []}
        title={source.title}
      />
    </li>
  );
};

const partLabel = (source: RetrievedSource): null | string => {
  if (source.section) {
    return source.section;
  }

  return typeof source.chunkIndex === 'number'
    ? `#${source.chunkIndex + 1}`
    : null;
};

const DocumentSnippets = ({
  id,
  sources,
}: {
  id: string;
  sources: readonly RetrievedSource[];
}) => (
  <ul
    className="mt-2 space-y-2 text-xs leading-relaxed text-muted-foreground"
    id={id}
  >
    {sources.map((source) => {
      const label = partLabel(source);

      return (
        <li
          className="whitespace-pre-wrap border-l border-border/70 pl-2"
          key={source.id}
        >
          {label === null ? null : (
            <p className="font-medium text-foreground/80">{label}</p>
          )}
          <MessageResponse
            className="h-auto"
            components={sourceMarkdownComponents}
            controls={false}
            disallowedElements={disallowedSourceMarkdownElements}
            mode="static"
            plugins={sourceMarkdownPlugins}
          >
            {source.snippet ?? ''}
          </MessageResponse>
        </li>
      );
    })}
  </ul>
);

// Several chunks of one document: the title once, its sections, and every
// chunk's snippet on expansion.
const SourceDocumentCard = ({
  citeRef,
  focused,
  sources,
}: CardCitation & {
  sources: readonly [RetrievedSource, ...RetrievedSource[]];
}) => {
  const [expanded, setExpanded] = useState(false);
  const snippetsId = useId();
  const [first] = sources;
  const sections = groupSections(sources);
  const withSnippets = sources.filter(
    (source) => (source.snippet ?? '').length > 0,
  );
  const hasSnippets = withSnippets.length > 0;
  const heading = (
    <>
      <SourceMeta
        citeRef={citeRef}
        source={first}
      />
      <SourceTitle
        expandable={hasSnippets}
        expanded={expanded}
        title={first.title}
      />
    </>
  );

  return (
    <li
      className={cardClass(focused)}
      data-ref={citeRef}
      data-testid="source-document"
    >
      {hasSnippets ? (
        <button
          aria-controls={expanded ? snippetsId : undefined}
          aria-expanded={expanded}
          className={CARD_TOGGLE_CLASS}
          onClick={() => {
            setExpanded((current) => !current);
          }}
          type="button"
        >
          {heading}
        </button>
      ) : (
        <div>{heading}</div>
      )}
      {sections.length > 0 ? (
        <ul
          aria-label={t('sources.sections')}
          className="mt-1.5 flex flex-wrap gap-1"
        >
          {sections.map((section) => (
            <li
              className="rounded-md border border-border/70 px-1.5 py-0.5 text-xs text-muted-foreground"
              key={section}
            >
              {section}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-1.5 text-xs text-muted-foreground">
          {`${sources.length} ${t('sources.partMany')}`}
        </p>
      )}
      {expanded ? (
        <DocumentSnippets
          id={snippetsId}
          sources={withSnippets}
        />
      ) : null}
      <OutdatedNote source={first} />
      <SourceLinks
        links={groupLinks(sources)}
        title={first.title}
      />
    </li>
  );
};

// A compact toggle for the message footer; the cards open below the footer.
export const SourcesToggle = ({
  count,
  onToggle,
  open,
  panelId,
}: {
  count: number;
  onToggle: () => void;
  open: boolean;
  panelId: string;
}) => (
  <button
    aria-controls={open ? panelId : undefined}
    aria-expanded={open}
    aria-label={open ? t('sources.hide') : t('sources.show')}
    className="inline-flex min-h-11 items-center gap-1.5 rounded-md px-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-8"
    data-testid="message-sources"
    onClick={onToggle}
    type="button"
  >
    <BookOpenText
      aria-hidden="true"
      className="size-3.5"
    />
    <span className="max-sm:hidden">{t('sources.title')}</span>
    <span className="tabular-nums text-muted-foreground/70">{count}</span>
    <ChevronRight
      aria-hidden="true"
      className={`size-3 transition-transform ${open ? 'rotate-90' : ''}`}
    />
  </button>
);

const prefersReducedMotion = (): boolean =>
  typeof matchMedia === 'function' &&
  matchMedia('(prefers-reduced-motion: reduce)').matches;

export const SourcesPanel = ({
  focusedRef = null,
  id,
  sources,
}: {
  focusedRef?: null | number;
  id: string;
  sources: readonly RetrievedSource[];
}) => {
  const panelRef = useRef<HTMLElement>(null);

  // A citation chip opened the panel: bring its card into view.
  useEffect(() => {
    if (focusedRef === null) {
      return;
    }

    panelRef.current
      ?.querySelector(`[data-ref="${CSS.escape(String(focusedRef))}"]`)
      ?.scrollIntoView({
        behavior: prefersReducedMotion() ? 'auto' : 'smooth',
        block: 'nearest',
      });
  }, [focusedRef]);

  return (
    <section
      aria-label={t('sources.title')}
      className="mt-2 motion-safe:animate-in motion-safe:fade-in-0"
      data-testid="source-panel"
      id={id}
      ref={panelRef}
    >
      <ul className="grid items-start gap-2 sm:grid-cols-2">
        {groupSources(sources).map(({ key, sources: grouped }) => {
          const [first, ...rest] = grouped;

          if (first === undefined) {
            return null;
          }

          const citeRef = groupRef(grouped);
          const focused = citeRef !== undefined && citeRef === focusedRef;

          return rest.length === 0 ? (
            <SourceCard
              citeRef={citeRef}
              focused={focused}
              key={`${first.kind}:${first.id}`}
              source={first}
            />
          ) : (
            <SourceDocumentCard
              citeRef={citeRef}
              focused={focused}
              key={key}
              sources={[first, ...rest]}
            />
          );
        })}
      </ul>
    </section>
  );
};

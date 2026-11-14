'use client';

import { type ReactNode, useMemo, useState } from 'react';

import type { RetrievedSource } from '@/lib/api-types';

import {
  CitationContext,
  type CitationState,
  useCitations,
} from '@/lib/citation-context';
import { sourceForRef } from '@/lib/citations';
import { t } from '@/lib/i18n';

// One per answer: its `[N]` chips and its footer share the sources panel.
export const CitationProvider = ({
  children,
  sources,
}: {
  children: ReactNode;
  sources: readonly RetrievedSource[];
}) => {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [focusedRef, setFocusedRef] = useState<null | number>(null);
  const value = useMemo<CitationState>(
    () => ({
      focusedRef,
      openSource: (ref) => {
        setSourcesOpen(true);
        setFocusedRef(ref);
      },
      setSourcesOpen: (open) => {
        setSourcesOpen(open);
        if (!open) {
          setFocusedRef(null);
        }
      },
      sources,
      sourcesOpen,
    }),
    [focusedRef, sources, sourcesOpen],
  );

  return <CitationContext value={value}>{children}</CitationContext>;
};

// The rendered `[N]` marker. Answers whose sources carry no numbers (older or
// Discord answers) keep the literal text; a number the panel cannot show is dropped.
export const CitationRef = ({
  'data-ref': rawRef,
}: {
  'data-ref'?: string;
}) => {
  const citations = useCitations();
  const ref = Number(rawRef);
  const source =
    citations === null ? undefined : sourceForRef(citations.sources, ref);

  if (citations === null || !Number.isSafeInteger(ref)) {
    return <>{`[${rawRef ?? ''}]`}</>;
  }

  if (source === undefined) {
    const numbered = citations.sources.some((item) => item.ref !== undefined);

    return <>{numbered ? null : `[${ref}]`}</>;
  }

  return (
    <button
      aria-label={`${t('sources.cite')} ${ref}: ${source.title}`}
      className="relative ml-0.5 inline-flex h-4 min-w-4 -translate-y-0.5 after:absolute after:-inset-2 after:content-[''] items-center justify-center rounded-full bg-muted px-1 align-middle text-[10px] font-medium leading-none tabular-nums text-muted-foreground transition-colors hover:bg-primary/15 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      data-testid="citation"
      onClick={() => {
        citations.openSource(ref);
      }}
      title={source.title}
      type="button"
    >
      {ref}
    </button>
  );
};

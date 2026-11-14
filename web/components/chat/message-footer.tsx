'use client';

import { Clock3 } from 'lucide-react';
import { posthog } from 'posthog-js';
import { type ReactNode, useId, useRef, useState } from 'react';

import type { MyUIMessage, RetrievedSource } from '@/lib/api-types';

import { SourcesPanel, SourcesToggle } from '@/components/chat/source-cards';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import { useCitations } from '@/lib/citation-context';
import { formatThroughput } from '@/lib/diagnostics';
import { formatDuration } from '@/lib/duration';
import { formatSpanLabel, t } from '@/lib/i18n';
import { sourceGroupCount } from '@/lib/source-groups';

type Diagnostics = NonNullable<MyUIMessage['metadata']>['diagnostics'];
type Timing = NonNullable<MyUIMessage['metadata']>['timing'];

const FOOTNOTE_CLASS =
  'inline-flex min-h-8 px-2 items-center gap-1 text-xs tabular-nums text-muted-foreground/70';

const formatMs = (ms: null | number | undefined): string =>
  typeof ms === 'number' ? formatDuration(ms) : '—';

const DiagnosticsRow = ({
  label,
  value,
  valueClassName = 'shrink-0 tabular-nums',
}: {
  label: string;
  value: string;
  valueClassName?: string;
}) => (
  <div className="flex items-center justify-between gap-6">
    <span className="min-w-0 truncate text-muted-foreground">{label}</span>
    <span className={valueClassName}>{value}</span>
  </div>
);

const DiagnosticsGroup = ({ children }: { children: ReactNode }) => (
  <div className="flex flex-col gap-1 border-t border-border/60 pt-2 first:border-t-0 first:pt-0">
    {children}
  </div>
);

const CostDiagnosticsRow = ({
  cost,
}: {
  cost: NonNullable<Diagnostics>['cost'];
}) =>
  cost === undefined ? null : (
    <DiagnosticsRow
      label={t('diagnostics.cost')}
      value={`$${cost.totalUsd.toFixed(6)}`}
    />
  );

const DiagnosticsCard = ({
  diagnostics,
  inferenceModel,
  traceId,
}: {
  diagnostics: NonNullable<Diagnostics>;
  inferenceModel?: string;
  traceId?: string;
}) => {
  const spans = Object.entries(diagnostics.spans ?? {});
  const { tokens } = diagnostics;
  const throughput = formatThroughput(diagnostics);
  const hasRetrievalShape =
    typeof diagnostics.candidateCount === 'number' ||
    typeof diagnostics.topDistance === 'number';

  return (
    <div className="flex flex-col gap-2 text-xs">
      <p className="font-medium text-foreground">{t('diagnostics.title')}</p>
      {inferenceModel || traceId ? (
        <DiagnosticsGroup>
          {inferenceModel ? (
            <DiagnosticsRow
              label={t('diagnostics.model')}
              value={inferenceModel}
            />
          ) : null}
          {traceId ? (
            <DiagnosticsRow
              label={t('diagnostics.traceId')}
              value={traceId}
              valueClassName="max-w-44 break-all text-right font-mono text-[11px] leading-snug"
            />
          ) : null}
        </DiagnosticsGroup>
      ) : null}
      <DiagnosticsGroup>
        {typeof diagnostics.thinkingMs === 'number' ? (
          <DiagnosticsRow
            label={t('diagnostics.thinking')}
            value={formatMs(diagnostics.thinkingMs)}
          />
        ) : null}
        <DiagnosticsRow
          label={t('diagnostics.serverTotal')}
          value={formatMs(diagnostics.serverTotalMs)}
        />
        <DiagnosticsRow
          label={t('diagnostics.serverFirstByte')}
          value={formatMs(diagnostics.serverTtftMs)}
        />
      </DiagnosticsGroup>
      {spans.length === 0 ? null : (
        <DiagnosticsGroup>
          <p className="text-muted-foreground/70">{t('diagnostics.stages')}</p>
          {spans.map(([name, ms]) => (
            <DiagnosticsRow
              key={name}
              label={formatSpanLabel(name)}
              value={formatMs(ms)}
            />
          ))}
        </DiagnosticsGroup>
      )}
      {hasRetrievalShape ? (
        <DiagnosticsGroup>
          <DiagnosticsRow
            label={t('diagnostics.candidates')}
            value={
              typeof diagnostics.candidateCount === 'number'
                ? String(diagnostics.candidateCount)
                : '—'
            }
          />
          <DiagnosticsRow
            label={t('diagnostics.topDistance')}
            value={
              typeof diagnostics.topDistance === 'number'
                ? diagnostics.topDistance.toFixed(4)
                : '—'
            }
          />
        </DiagnosticsGroup>
      ) : null}
      {tokens === undefined ? null : (
        <DiagnosticsGroup>
          <DiagnosticsRow
            label={t('diagnostics.tokensTotal')}
            value={String(tokens.total)}
          />
          <DiagnosticsRow
            label={t('diagnostics.tokensInput')}
            value={String(tokens.input)}
          />
          <DiagnosticsRow
            label={t('diagnostics.tokensOutput')}
            value={String(tokens.output)}
          />
          {throughput === null ? null : (
            <DiagnosticsRow
              label={t('diagnostics.throughput')}
              value={throughput}
            />
          )}
          <CostDiagnosticsRow cost={diagnostics.cost} />
        </DiagnosticsGroup>
      )}
    </div>
  );
};

// The total only; the time to the first token is in the label and the detail.
const TimingSummary = ({ timing }: { timing: NonNullable<Timing> }) => (
  <>
    <Clock3
      aria-hidden="true"
      className="size-3"
    />
    <span>{formatDuration(timing.totalMs)}</span>
  </>
);

const timingLabel = (timing: NonNullable<Timing>): string =>
  timing.ttftMs === null
    ? formatDuration(timing.totalMs)
    : `${formatDuration(timing.totalMs)}, ${t('thread.firstToken')} ${formatDuration(timing.ttftMs)}`;

const MessageTiming = ({
  diagnostics,
  inferenceModel,
  responseId,
  timing,
}: {
  diagnostics: Diagnostics;
  inferenceModel?: string;
  responseId?: string;
  timing: Timing;
}) => {
  const diagnosticsContentRef = useRef<HTMLDivElement>(null);

  if (timing === undefined) {
    return null;
  }

  if (diagnostics === undefined || Object.keys(diagnostics).length === 0) {
    return (
      <div
        className={FOOTNOTE_CLASS}
        data-testid="message-timing"
        title={timingLabel(timing)}
      >
        <TimingSummary timing={timing} />
      </div>
    );
  }

  // The HoverCard detail is pointer/focus-only, so name the trigger with purpose + summary.
  const triggerLabel = `${t('diagnostics.title')}: ${timingLabel(timing)}`;

  return (
    <HoverCard
      closeDelay={80}
      openDelay={120}
    >
      <HoverCardTrigger asChild>
        <button
          aria-label={triggerLabel}
          className={`${FOOTNOTE_CLASS} cursor-help rounded-md hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring`}
          data-testid="message-timing"
          onKeyDown={(event) => {
            const content = diagnosticsContentRef.current;
            if (content === null) {
              return;
            }

            switch (event.key) {
              case 'ArrowDown':
                content.scrollBy({ top: content.clientHeight / 4 });
                break;
              case 'ArrowUp':
                content.scrollBy({ top: -content.clientHeight / 4 });
                break;
              case 'End':
                content.scrollTo({ top: content.scrollHeight });
                break;
              case 'Home':
                content.scrollTo({ top: 0 });
                break;
              case 'PageDown':
                content.scrollBy({ top: content.clientHeight });
                break;
              case 'PageUp':
                content.scrollBy({ top: -content.clientHeight });
                break;
              default:
                return;
            }
            event.preventDefault();
          }}
          type="button"
        >
          <TimingSummary timing={timing} />
        </button>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        className="max-h-(--radix-hover-card-content-available-height) w-auto min-w-[min(14rem,var(--radix-hover-card-content-available-width))] max-w-[min(20rem,var(--radix-hover-card-content-available-width))] overflow-y-auto overscroll-contain"
        collisionPadding={16}
        ref={diagnosticsContentRef}
      >
        <DiagnosticsCard
          diagnostics={diagnostics}
          inferenceModel={inferenceModel}
          traceId={responseId}
        />
      </HoverCardContent>
    </HoverCard>
  );
};

export const MessageFooter = ({
  actions,
  diagnostics,
  inferenceModel,
  responseId,
  sources,
  timing,
}: {
  actions: ReactNode;
  diagnostics: Diagnostics;
  inferenceModel?: string;
  responseId?: string;
  sources: readonly RetrievedSource[];
  timing: Timing;
}) => {
  // Inside an answer the panel is shared with its `[N]` chips.
  const citations = useCitations();
  const [localOpen, setLocalOpen] = useState(false);
  const sourcesOpen = citations?.sourcesOpen ?? localOpen;
  const setSourcesOpen = citations?.setSourcesOpen ?? setLocalOpen;
  const panelId = useId();
  const count = sourceGroupCount(sources);
  const hasMeta = count > 0 || timing !== undefined;

  if (!hasMeta && (actions === undefined || actions === null)) {
    return null;
  }

  return (
    <div
      className="mt-2"
      data-testid="message-footer"
    >
      <div className="flex flex-wrap items-center gap-1">
        {actions}
        {hasMeta ? (
          // Hidden when the actions render nothing and it would lead the row.
          <span
            aria-hidden="true"
            className="mx-1 h-4 w-px bg-border first:hidden"
          />
        ) : null}
        {count > 0 ? (
          <SourcesToggle
            count={count}
            onToggle={() => {
              if (!sourcesOpen) {
                /* eslint-disable camelcase -- PostHog event properties are snake_case. */
                posthog.capture('sources_opened', {
                  response_id: responseId,
                  source_count: count,
                });
                /* eslint-enable camelcase -- end of PostHog snake_case properties. */
              }
              setSourcesOpen(!sourcesOpen);
            }}
            open={sourcesOpen}
            panelId={panelId}
          />
        ) : null}
        <MessageTiming
          diagnostics={diagnostics}
          inferenceModel={inferenceModel}
          responseId={responseId}
          timing={timing}
        />
      </div>
      {sourcesOpen && count > 0 ? (
        <SourcesPanel
          focusedRef={citations?.focusedRef ?? null}
          id={panelId}
          sources={sources}
        />
      ) : null}
    </div>
  );
};

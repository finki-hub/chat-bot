import { Clock3 } from 'lucide-react';
import { type ReactNode, useRef } from 'react';

import type { ErrorNotice, MyUIMessage, StatusPart } from '@/lib/api-types';

import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { ReasoningDisclosure } from '@/components/chat/reasoning-disclosure';
import { SearchStatus } from '@/components/chat/search-status';
import { SearchStepper } from '@/components/chat/search-stepper';
import { SourceCards } from '@/components/chat/source-cards';
import { TypingIndicator } from '@/components/chat/typing-indicator';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import { formatThroughput } from '@/lib/diagnostics';
import { formatDuration } from '@/lib/duration';
import { formatSpanLabel, t } from '@/lib/i18n';
import { reasoningParts, textParts } from '@/lib/message-parts';
import { usePresence } from '@/lib/use-presence';
import { useSearchStage } from '@/lib/use-search-stage';

export type AssistantMessageProps = {
  actions?: ReactNode;
  errorPart?: ErrorNotice;
  message: MyUIMessage;
  onManageCredentials?: () => void;
  onRetry?: () => void;
  onWait?: () => void;
  pending?: boolean;
  statusPart?: StatusPart;
};

type Diagnostics = NonNullable<MyUIMessage['metadata']>['diagnostics'];
type Timing = NonNullable<MyUIMessage['metadata']>['timing'];

const FOOTNOTE_CLASS =
  'mt-1 inline-flex items-center gap-1 text-xs tabular-nums text-muted-foreground/70';

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

const TimingSummary = ({ timing }: { timing: NonNullable<Timing> }) => (
  <>
    <Clock3
      aria-hidden="true"
      className="size-3"
    />
    <span>
      {formatDuration(timing.totalMs)}
      {timing.ttftMs === null
        ? null
        : ` · ${t('thread.firstToken')} ${formatDuration(timing.ttftMs)}`}
    </span>
  </>
);

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
      >
        <TimingSummary timing={timing} />
      </div>
    );
  }

  const ttftSuffix =
    timing.ttftMs === null
      ? ''
      : `, ${t('thread.firstToken')} ${formatDuration(timing.ttftMs)}`;
  // The HoverCard detail is pointer/focus-only, so name the trigger with purpose + summary.
  const triggerLabel = `${t('diagnostics.title')}: ${formatDuration(timing.totalMs)}${ttftSuffix}`;

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
        className="max-h-[var(--radix-hover-card-content-available-height)] w-auto min-w-[min(14rem,var(--radix-hover-card-content-available-width))] max-w-[min(20rem,var(--radix-hover-card-content-available-width))] overflow-y-auto overscroll-contain"
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

const formatResetTime = (resetsAt: string | undefined): string | undefined => {
  if (resetsAt === undefined) {
    return undefined;
  }

  // eslint-disable-next-line unicorn/prefer-temporal -- browser support is required here.
  const resetDate = new Date(resetsAt);
  if (Number.isNaN(resetDate.getTime())) {
    return undefined;
  }

  return new Intl.DateTimeFormat('mk-MK', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(resetDate);
};

const SponsoredQuotaError = ({
  errorPart,
  onManageCredentials,
  onWait,
}: {
  errorPart: ErrorNotice;
  onManageCredentials?: () => void;
  onWait?: () => void;
}) => {
  const resetTime = formatResetTime(errorPart.resets_at);

  return (
    <div className="flex flex-col gap-2">
      <p className="text-destructive">{t('error.sponsoredQuota')}</p>
      {resetTime === undefined ? null : (
        <p className="text-muted-foreground">
          {t('error.sponsoredQuotaReset')} {resetTime}.
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {onManageCredentials ? (
          <button
            className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onClick={onManageCredentials}
            type="button"
          >
            {t('error.manageCredentials')}
          </button>
        ) : null}
        {onWait ? (
          <button
            className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onClick={onWait}
            type="button"
          >
            {t('error.sponsoredWait')}
          </button>
        ) : null}
      </div>
    </div>
  );
};

export const MessageError = ({
  errorPart,
  onManageCredentials,
  onRetry,
  onWait,
}: {
  errorPart: ErrorNotice;
  onManageCredentials?: () => void;
  onRetry?: () => void;
  onWait?: () => void;
}) => {
  let content: ReactNode;

  switch (errorPart.code) {
    case 'credential_required':
      content = (
        <div className="flex flex-col gap-2">
          <p className="text-destructive">{t('error.credentialRequired')}</p>
          {onManageCredentials ? (
            <button
              className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={onManageCredentials}
              type="button"
            >
              {t('error.manageCredentials')}
            </button>
          ) : null}
        </div>
      );
      break;
    case 'free_quota_exhausted':
      content = (
        <SponsoredQuotaError
          errorPart={errorPart}
          onManageCredentials={onManageCredentials}
          onWait={onWait}
        />
      );
      break;
    case 'free_tier_unavailable':
      content = (
        <p className="text-destructive">{t('error.sponsoredUnavailable')}</p>
      );
      break;
    case 'interrupted':
      content = (
        <p className="text-muted-foreground">{t('error.interrupted')}</p>
      );
      break;
    case 'no_answer':
      content = <p className="text-destructive">{t('error.noAnswer')}</p>;
      break;
    case 'sponsored_request_in_progress':
      content = (
        <p className="text-destructive">{t('error.sponsoredConcurrent')}</p>
      );
      break;
    default:
      content = (
        <div className="flex flex-col gap-2">
          <p className="text-destructive">{t('error.description')}</p>
          {onRetry ? (
            <button
              className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={onRetry}
              type="button"
            >
              {t('error.retry')}
            </button>
          ) : null}
        </div>
      );
      break;
  }

  return (
    <div
      className="mt-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm"
      role="alert"
    >
      {content}
    </div>
  );
};

// Keep in sync with the `duration-300` exit transition on the stepper wrapper.
const STEP_EXIT_MS = 300;

const AssistantMessageStatus = ({
  errorPart,
  pending,
  reasoningText,
  statusPart,
  text,
}: {
  errorPart?: ErrorNotice;
  pending?: boolean;
  reasoningText: string;
  statusPart?: StatusPart;
  text: null | string;
}) => {
  const stages = useSearchStage({
    pending,
    reasoningActive: reasoningText.length > 0,
    stage: statusPart?.stage,
    statusActive: Boolean(statusPart),
    text,
  });

  const showStepper =
    stages.length > 0 && text === null && (pending ?? Boolean(statusPart));
  const stepper = usePresence(showStepper, STEP_EXIT_MS);
  const showChip = Boolean(statusPart) && !text && !statusPart?.stage;
  const showDots =
    Boolean(pending) &&
    !text &&
    !statusPart &&
    !errorPart &&
    reasoningText.length === 0 &&
    stages.length === 0;

  if (stepper.mounted) {
    // Collapse + fade the steps on exit (grid-rows 1fr→0fr squashes them
    // upward) instead of unmounting abruptly when the answer arrives.
    return (
      <div
        aria-hidden={stepper.exiting}
        className={`grid transition-[grid-template-rows,opacity] duration-300 ease-out motion-reduce:transition-none ${
          stepper.exiting
            ? 'grid-rows-[0fr] opacity-0'
            : 'grid-rows-[1fr] opacity-100'
        }`}
      >
        <div className="min-h-0 overflow-hidden">
          <SearchStepper stages={stages} />
        </div>
      </div>
    );
  }

  if (showChip && statusPart) {
    return (
      <div data-testid="search-status-wrapper">
        <SearchStatus
          label={statusPart.label}
          tool={statusPart.tool}
        />
      </div>
    );
  }

  if (showDots) {
    return <TypingIndicator />;
  }

  return null;
};
export const AssistantMessage = ({
  actions,
  errorPart,
  message,
  onManageCredentials,
  onRetry,
  onWait,
  pending,
  statusPart,
}: AssistantMessageProps) => {
  const parts = textParts(message);
  const reasoningText = reasoningParts(message)
    .map((part) => part.text)
    .join('');
  // A text part that is still the FIRST part while a stage-less tool/chip status
  // is active is the model's pre-tool preamble (the backend discards it via a
  // `reset` once the real answer begins); suppress it so it does not render and
  // then get swapped for the answer. A retrieval `stage` status, by contrast,
  // stays active through generation in the pipeline path (no reset is sent), so
  // it must NOT suppress the real answer streaming under it.
  const inPreamble =
    Boolean(pending) &&
    Boolean(statusPart) &&
    statusPart?.stage === undefined &&
    parts.length <= 1;
  const text = inPreamble ? null : (parts.at(-1)?.text ?? null);
  const answerVisible = Boolean(text);
  const reasoningStreaming =
    Boolean(pending) && reasoningText.length > 0 && text === null;

  const timing = message.metadata?.timing;
  const diagnostics = message.metadata?.diagnostics;
  const inferenceModel = message.metadata?.inferenceModel;
  const sources = message.metadata?.sources ?? [];
  const responseId = message.metadata?.responseId;

  return (
    <Message from="assistant">
      <MessageContent>
        {/* Once the answer is visible, reasoning collapses to a toggle above it. */}
        {answerVisible && reasoningText.length > 0 ? (
          <ReasoningDisclosure
            streaming={reasoningStreaming}
            text={reasoningText}
          />
        ) : null}
        {text ? (
          <div data-testid="answer-text">
            <MessageResponse>{text}</MessageResponse>
          </div>
        ) : null}
        {/* No `pending` guard: MessageTiming renders nothing until `timing` is set
            on finish, so a still-streaming answer shows no footnote anyway. */}
        {text === null ? null : (
          <MessageTiming
            diagnostics={diagnostics}
            inferenceModel={inferenceModel}
            responseId={responseId}
            timing={timing}
          />
        )}
        {text === null ? null : <SourceCards sources={sources} />}
        <AssistantMessageStatus
          errorPart={errorPart}
          pending={pending}
          reasoningText={reasoningText}
          statusPart={statusPart}
          text={text}
        />
        {/* While still streaming (no answer yet), reasoning streams in BELOW the
            step list so it grows downward instead of displacing the steps. */}
        {!answerVisible && reasoningText.length > 0 ? (
          <ReasoningDisclosure
            streaming={reasoningStreaming}
            text={reasoningText}
          />
        ) : null}
        {errorPart ? (
          <MessageError
            errorPart={errorPart}
            onManageCredentials={onManageCredentials}
            onRetry={onRetry}
            onWait={onWait}
          />
        ) : null}
        {answerVisible && actions !== undefined && actions !== null ? (
          <div className="mt-2">{actions}</div>
        ) : null}
      </MessageContent>
    </Message>
  );
};

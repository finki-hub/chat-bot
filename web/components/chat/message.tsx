import type { ReactNode } from 'react';

import { Clock3 } from 'lucide-react';

import type { MyUIMessage } from '@/lib/api-types';

import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { ElapsedTimer } from '@/components/chat/elapsed-timer';
import { ReasoningDisclosure } from '@/components/chat/reasoning-disclosure';
import { SearchStatus } from '@/components/chat/search-status';
import { TypingIndicator } from '@/components/chat/typing-indicator';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import { formatDuration } from '@/lib/duration';
import { formatSpanLabel, t } from '@/lib/i18n';
import { reasoningParts, textParts } from '@/lib/message-parts';

export type AssistantMessageProps = {
  actions?: ReactNode;
  errorPart?: { code: string; message: string };
  message: MyUIMessage;
  onRetry?: () => void;
  pending?: boolean;
  statusPart?: { label: string; tool?: string };
  streamStartedAt?: null | number;
};

type Diagnostics = NonNullable<MyUIMessage['metadata']>['diagnostics'];
type Timing = NonNullable<MyUIMessage['metadata']>['timing'];

const FOOTNOTE_CLASS =
  'mt-1 inline-flex items-center gap-1 text-xs tabular-nums text-muted-foreground/70';

const formatMs = (ms: null | number | undefined): string =>
  typeof ms === 'number' ? formatDuration(ms) : '—';

const DiagnosticsRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between gap-6">
    <span className="min-w-0 truncate text-muted-foreground">{label}</span>
    <span className="shrink-0 tabular-nums">{value}</span>
  </div>
);

const DiagnosticsGroup = ({ children }: { children: ReactNode }) => (
  <div className="flex flex-col gap-1 border-t border-border/60 pt-2 first:border-t-0 first:pt-0">
    {children}
  </div>
);

const DiagnosticsCard = ({
  diagnostics,
}: {
  diagnostics: NonNullable<Diagnostics>;
}) => {
  const spans = Object.entries(diagnostics.spans ?? {});
  const { tokens } = diagnostics;
  const hasRetrievalShape =
    typeof diagnostics.candidateCount === 'number' ||
    typeof diagnostics.topDistance === 'number';

  return (
    <div className="flex flex-col gap-2 text-xs">
      <p className="font-medium text-foreground">{t('diagnostics.title')}</p>
      <DiagnosticsGroup>
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
  timing,
}: {
  diagnostics: Diagnostics;
  timing: Timing;
}) => {
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
          className={`${FOOTNOTE_CLASS} cursor-help hover:text-muted-foreground`}
          data-testid="message-timing"
          type="button"
        >
          <TimingSummary timing={timing} />
        </button>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        className="w-auto min-w-56 max-w-xs"
      >
        <DiagnosticsCard diagnostics={diagnostics} />
      </HoverCardContent>
    </HoverCard>
  );
};

const MessageError = ({
  errorPart,
  onRetry,
}: {
  errorPart: { code: string; message: string };
  onRetry?: () => void;
}) => (
  <div
    className="mt-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm"
    role="alert"
  >
    {errorPart.code === 'interrupted' ? (
      <p className="text-muted-foreground">{t('error.interrupted')}</p>
    ) : (
      <div className="flex flex-col gap-2">
        <p className="text-destructive">{errorPart.message}</p>
        {onRetry ? (
          <button
            className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted"
            onClick={onRetry}
            type="button"
          >
            {t('error.retry')}
          </button>
        ) : null}
      </div>
    )}
  </div>
);

export const AssistantMessage = ({
  actions,
  errorPart,
  message,
  onRetry,
  pending,
  statusPart,
  streamStartedAt,
}: AssistantMessageProps) => {
  const parts = textParts(message);
  const reasoningText = reasoningParts(message)
    .map((part) => part.text)
    .join('');
  // While streaming with a tool status active, a text part that is still the
  // FIRST part is the model's pre-tool preamble (the backend discards it via a
  // `reset` once the real answer begins). Suppress it so it does not render and
  // then get swapped for the answer once the answer part starts.
  const inPreamble =
    Boolean(pending) && Boolean(statusPart) && parts.length <= 1;
  const text = inPreamble ? null : (parts.at(-1)?.text ?? null);
  const showChip = Boolean(statusPart) && !text;
  const showDots = Boolean(pending) && !text && !statusPart && !errorPart;
  const timing = message.metadata?.timing;
  const diagnostics = message.metadata?.diagnostics;
  const liveTimer =
    typeof streamStartedAt === 'number' ? (
      <ElapsedTimer startedAt={streamStartedAt} />
    ) : null;

  return (
    <Message from="assistant">
      <MessageContent>
        {reasoningText ? <ReasoningDisclosure text={reasoningText} /> : null}
        {text ? (
          <div data-testid="answer-text">
            <MessageResponse>{text}</MessageResponse>
          </div>
        ) : null}
        {pending || text === null ? null : (
          <MessageTiming
            diagnostics={diagnostics}
            timing={timing}
          />
        )}
        {showChip && statusPart ? (
          <div className="flex items-center gap-2">
            <SearchStatus
              label={statusPart.label}
              tool={statusPart.tool}
            />
            {liveTimer}
          </div>
        ) : null}
        {showDots ? (
          <div className="flex items-center gap-2">
            <TypingIndicator />
            {liveTimer}
          </div>
        ) : null}
        {errorPart ? (
          <MessageError
            errorPart={errorPart}
            onRetry={onRetry}
          />
        ) : null}
        {actions === undefined || actions === null ? null : (
          <div className="mt-2">{actions}</div>
        )}
      </MessageContent>
    </Message>
  );
};

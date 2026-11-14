import type { DynamicToolUIPart } from 'ai';

import { TriangleAlert } from 'lucide-react';
import { type ReactNode } from 'react';

import type { ErrorNotice, MyUIMessage, StatusPart } from '@/lib/api-types';

import { Message, MessageContent } from '@/components/ai-elements/message';
import { ActivityLine } from '@/components/chat/activity-line';
import { AnswerMarkdown } from '@/components/chat/answer-markdown';
import { CitationProvider } from '@/components/chat/citations';
import { MessageFooter } from '@/components/chat/message-footer';
import { ReasoningDisclosure } from '@/components/chat/reasoning-disclosure';
import { SearchStepper } from '@/components/chat/search-stepper';
import {
  type FinishedReasoning,
  ToolCall,
  ToolCallGroup,
} from '@/components/chat/tool-call';
import { Button } from '@/components/ui/button';
import { t } from '@/lib/i18n';
import {
  hasToolCalls,
  isTextPart,
  isToolCallPart,
  joinReasoning,
  textParts,
  toolCallParts,
} from '@/lib/message-parts';
import { isSearchTool, searchToolLabel } from '@/lib/search-status';
import { supersededToolCalls, toolCallStatus } from '@/lib/tool-calls';
import { usePresence } from '@/lib/use-presence';
import { useSearchStage } from '@/lib/use-search-stage';

export type AssistantMessageProps = {
  actions?: ReactNode;
  errorPart?: ErrorNotice;
  message: MyUIMessage;
  onManageCredentials?: (returnFocusTarget: HTMLElement) => void;
  onRetry?: () => void;
  pending?: boolean;
  retryPending?: boolean;
  statusPart?: StatusPart;
};

const MACEDONIAN_SHORT_MONTHS = [
  'јан.',
  'фев.',
  'мар.',
  'апр.',
  'мај',
  'јун.',
  'јул.',
  'авг.',
  'септ.',
  'окт.',
  'ноем.',
  'дек.',
] as const;
const SKOPJE_RESET_TIME_FORMATTER = new Intl.DateTimeFormat('en-US', {
  day: 'numeric',
  hour: '2-digit',
  hourCycle: 'h23',
  minute: '2-digit',
  month: 'numeric',
  timeZone: 'Europe/Skopje',
  year: 'numeric',
});

const formatResetTime = (resetsAt: string | undefined): string | undefined => {
  if (resetsAt === undefined) {
    return undefined;
  }

  // eslint-disable-next-line unicorn/prefer-temporal -- browser support is required here.
  const resetDate = new Date(resetsAt);
  if (Number.isNaN(resetDate.getTime())) {
    return undefined;
  }

  const parts = new Map(
    SKOPJE_RESET_TIME_FORMATTER.formatToParts(resetDate).map((part) => [
      part.type,
      part.value,
    ]),
  );
  const day = parts.get('day');
  if (day === undefined) {
    return undefined;
  }

  const hours = parts.get('hour');
  if (hours === undefined) {
    return undefined;
  }

  const minutes = parts.get('minute');
  if (minutes === undefined) {
    return undefined;
  }

  const month = MACEDONIAN_SHORT_MONTHS.at(
    Number.parseInt(parts.get('month') ?? '', 10) - 1,
  );
  if (month === undefined) {
    return undefined;
  }

  const year = parts.get('year');
  return year === undefined
    ? undefined
    : `${day} ${month} ${year}, ${hours}:${minutes}`;
};

const SponsoredQuotaError = ({
  errorPart,
  onManageCredentials,
}: {
  errorPart: ErrorNotice;
  onManageCredentials?: (returnFocusTarget: HTMLElement) => void;
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
      {onManageCredentials ? (
        <Button
          className="min-h-11 self-start sm:pointer-fine:min-h-8"
          onClick={(event) => {
            onManageCredentials(event.currentTarget);
          }}
          size="sm"
          type="button"
        >
          {t('error.manageCredentials')}
        </Button>
      ) : null}
    </div>
  );
};

export const MessageError = ({
  errorPart,
  onManageCredentials,
  onRetry,
  retryPending,
}: {
  errorPart: ErrorNotice;
  onManageCredentials?: (returnFocusTarget: HTMLElement) => void;
  onRetry?: () => void;
  retryPending?: boolean;
}) => {
  // The provider failed after part of the answer streamed: the text above is
  // kept, so this warns that it may be incomplete rather than reporting a total
  // failure. (A user's Stop sets no error at all.)
  if (errorPart.code === 'interrupted') {
    return (
      <div
        className="mt-2 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-300"
        data-testid="message-interrupted"
        role="alert"
      >
        <TriangleAlert
          aria-hidden="true"
          className="mt-0.5 size-4 shrink-0"
        />
        {t('error.interrupted')}
      </div>
    );
  }

  let content: ReactNode;

  switch (errorPart.code) {
    case 'credential_required':
      content = (
        <div className="flex flex-col gap-2">
          <p className="text-destructive">{t('error.credentialRequired')}</p>
          {onManageCredentials ? (
            <button
              className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={(event) => {
                onManageCredentials(event.currentTarget);
              }}
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
        />
      );
      break;
    case 'free_tier_unavailable':
      content = (
        <p className="text-destructive">{t('error.sponsoredUnavailable')}</p>
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
              aria-busy={retryPending}
              className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              disabled={retryPending}
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
  awaitingLabel,
  awaitingModel = false,
  errorPart,
  pending,
  reasoningText,
  statusPart,
  text,
  toolCallsShown = false,
}: {
  awaitingLabel: string;
  awaitingModel?: boolean;
  errorPart?: ErrorNotice;
  pending?: boolean;
  reasoningText: string;
  statusPart?: StatusPart;
  text: null | string;
  toolCallsShown?: boolean;
}) => {
  const stages = useSearchStage({
    pending,
    reasoningActive: reasoningText.length > 0,
    stage: statusPart?.stage,
    statusActive: Boolean(statusPart),
    text,
  });

  // Tool-call messages show their progress inline as tool rows instead.
  const showStepper =
    !toolCallsShown &&
    stages.length > 0 &&
    text === null &&
    (pending ?? Boolean(statusPart));
  const stepper = usePresence(showStepper, STEP_EXIT_MS);
  const showChip =
    Boolean(statusPart) && !text && !statusPart?.stage && !toolCallsShown;
  const showDots =
    awaitingModel ||
    (!toolCallsShown &&
      Boolean(pending) &&
      !text &&
      !statusPart &&
      !errorPart &&
      reasoningText.length === 0 &&
      stages.length === 0);

  if (!toolCallsShown && stepper.mounted) {
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
    // The API's chip label is a first-person live message; show the tool's own.
    const label = searchToolLabel(statusPart.tool);

    return (
      <ActivityLine
        key={label}
        label={label}
      />
    );
  }

  return showDots ? (
    <ActivityLine
      key={awaitingLabel}
      label={awaitingLabel}
    />
  ) : null;
};

type AnswerSegment =
  | { key: string; part: DynamicToolUIPart; type: 'tool' }
  | { key: string; text: string; type: 'text' };

// Text parts carry no id, so each is keyed by the tool call it follows.
const answerSegments = (message: MyUIMessage): AnswerSegment[] => {
  const segments: AnswerSegment[] = [];
  let anchor = 'start';
  let textCount = 0;

  for (const part of message.parts) {
    if (isToolCallPart(part)) {
      anchor = part.toolCallId;
      textCount = 0;
      segments.push({ key: anchor, part, type: 'tool' });
    } else if (part.type === 'text' && part.text.length > 0) {
      textCount += 1;
      segments.push({
        key: `${anchor}:text:${textCount}`,
        text: part.text,
        type: 'text',
      });
    }
  }

  return segments;
};

type RenderSegment =
  AnswerSegment | { key: string; parts: DynamicToolUIPart[]; type: 'tools' };

// While streaming, the run still at the end is in progress and stays expanded;
// a run the model has written past folds as soon as the text after it starts,
// so finishing the message changes nothing.
const collapseToolRuns = (
  segments: AnswerSegment[],
  keepTrailing: boolean,
): RenderSegment[] => {
  const collapsed: RenderSegment[] = [];

  for (const segment of segments) {
    const previous = collapsed.at(-1);

    if (segment.type !== 'tool') {
      collapsed.push(segment);
    } else if (previous?.type === 'tools') {
      previous.parts.push(segment.part);
    } else if (previous?.type === 'tool') {
      collapsed[collapsed.length - 1] = {
        key: `${previous.key}:group`,
        parts: [previous.part, segment.part],
        type: 'tools',
      };
    } else {
      collapsed.push(segment);
    }
  }

  const last = collapsed.at(-1);

  if (keepTrailing && last?.type === 'tools') {
    collapsed.splice(
      -1,
      1,
      ...last.parts.map((part): AnswerSegment => ({
        key: part.toolCallId,
        part,
        type: 'tool',
      })),
    );
  }

  return collapsed;
};

// The answer is the first text after the last tool row; reasoning sits right
// above it, or at the end while there is no answer yet.
const reasoningSlot = (segments: readonly RenderSegment[]): number => {
  const lastTool = segments.findLastIndex((segment) => segment.type !== 'text');
  const answer = segments.findIndex(
    (segment, index) => index > lastTool && segment.type === 'text',
  );

  return answer === -1 ? segments.length : answer;
};

const SegmentView = ({
  className,
  pending,
  reasoning,
  segment,
  streamingTail,
  superseded,
}: {
  className: string | undefined;
  pending: boolean;
  reasoning: FinishedReasoning | undefined;
  segment: RenderSegment;
  streamingTail: boolean;
  superseded: ReadonlySet<string>;
}) => {
  if (segment.type === 'text') {
    return (
      <div
        className={className}
        data-testid="answer-text"
      >
        <AnswerMarkdown
          caret="block"
          isAnimating={streamingTail}
        >
          {segment.text}
        </AnswerMarkdown>
      </div>
    );
  }

  return segment.type === 'tool' ? (
    <ToolCall
      className={className}
      part={segment.part}
      running={pending && !superseded.has(segment.part.toolCallId)}
    />
  ) : (
    <ToolCallGroup
      className={`${className ?? ''} motion-safe:animate-in motion-safe:fade-in-0`}
      parts={segment.parts}
      reasoning={reasoning}
    />
  );
};

// Tool-call messages render every part in stream order: text written before,
// between and after the calls is all part of the answer. Runs of several calls
// fold into one summary row (see collapseToolRuns).
const InterleavedParts = ({
  finishedReasoning,
  message,
  pending,
  reasoning,
}: {
  finishedReasoning: FinishedReasoning | undefined;
  message: MyUIMessage;
  pending: boolean;
  reasoning: ReactNode;
}) => {
  const superseded = supersededToolCalls(message);
  const answer = answerSegments(message);
  const folded: RenderSegment[] =
    toolCallParts(message).length > 1
      ? collapseToolRuns(answer, pending)
      : answer;
  const candidate = reasoning === null ? -1 : reasoningSlot(folded);
  const previous = folded[candidate - 1];
  // A lone finished call before the answer folds too, so it can carry the
  // reasoning in one row instead of two.
  const segments =
    finishedReasoning !== undefined &&
    candidate > 0 &&
    candidate < folded.length &&
    previous?.type === 'tool'
      ? folded.with(candidate - 1, {
          key: `${previous.key}:group`,
          parts: [previous.part],
          type: 'tools',
        })
      : folded;
  // Finished reasoning right after a folded run joins that run's row.
  const mergeInto =
    finishedReasoning !== undefined &&
    candidate > 0 &&
    segments[candidate - 1]?.type === 'tools'
      ? candidate - 1
      : -1;
  const slot = mergeInto === -1 ? candidate : -1;
  const isText = (index: number): boolean => segments[index]?.type === 'text';
  // Extra space where tool rows give way to text (or back), so the answer does
  // not read as part of the last row; reasoning above the answer carries it.
  const spacing = (index: number): string | undefined =>
    index !== slot && index > 0 && isText(index - 1) !== isText(index)
      ? 'mt-2'
      : undefined;
  const items = segments.map((segment, index) => (
    <SegmentView
      className={spacing(index)}
      key={segment.key}
      pending={pending}
      reasoning={index === mergeInto ? finishedReasoning : undefined}
      segment={segment}
      streamingTail={pending && index === segments.length - 1}
      superseded={superseded}
    />
  ));

  if (slot !== -1) {
    items.splice(
      slot,
      0,
      <div
        className={slot > 0 && !isText(slot - 1) ? 'mt-2' : undefined}
        key="reasoning"
      >
        {reasoning}
      </div>,
    );
  }

  return (
    <div
      className="flex flex-col gap-2"
      data-testid="answer-parts"
    >
      {items}
    </div>
  );
};

const nonEmptyTexts = (parts: MyUIMessage['parts']): string[] =>
  parts
    .filter(isTextPart)
    .map((part) => part.text)
    .filter((text) => text.trim() !== '');

type AnswerState = {
  awaitingModel: boolean;
  text: null | string;
  withToolCalls: boolean;
};

const answerState = ({
  errorPart,
  message,
  pending,
  statusPart,
}: Pick<
  AssistantMessageProps,
  'errorPart' | 'message' | 'pending' | 'statusPart'
>): AnswerState => {
  if (!pending && hasToolCalls(message)) {
    return {
      awaitingModel: false,
      text: nonEmptyTexts(message.parts).at(-1) ?? null,
      withToolCalls: true,
    };
  }

  if (hasToolCalls(message)) {
    // While streaming, only text after the last call is the answer; earlier
    // segments are preamble the model wrote before searching.
    const answer = nonEmptyTexts(
      message.parts.slice(message.parts.findLastIndex(isToolCallPart) + 1),
    ).at(-1);
    const superseded = supersededToolCalls(message);
    const callsRunning = toolCallParts(message).some(
      (part) =>
        toolCallStatus(part, !superseded.has(part.toolCallId)) === 'running',
    );

    return {
      // Once the calls resolve, the model is writing its next turn.
      awaitingModel: !errorPart && answer === undefined && !callsRunning,
      text: answer ?? null,
      withToolCalls: true,
    };
  }

  const parts = textParts(message);

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

  return {
    awaitingModel: false,
    text: inPreamble ? null : (parts.at(-1)?.text ?? null),
    withToolCalls: false,
  };
};

// What the model is doing while nothing is streaming: after a search it is
// reading the sources, after data tools it is working through their output.
const awaitingActivityLabel = (message: MyUIMessage): string => {
  const parts = toolCallParts(message);

  if (parts.length === 0) {
    return t('activity.processing');
  }

  return parts.some(
    (part) => isSearchTool(part.toolName) && part.state === 'output-available',
  )
    ? t('activity.readingSources')
    : t('activity.readingResults');
};

export const AssistantMessage = ({
  actions,
  errorPart,
  message,
  onManageCredentials,
  onRetry,
  pending,
  retryPending,
  statusPart,
}: AssistantMessageProps) => {
  const { awaitingModel, text, withToolCalls } = answerState({
    errorPart,
    message,
    pending,
    statusPart,
  });
  const reasoningText = joinReasoning(message);
  const answerVisible = Boolean(text);
  // Answer text written between calls stays the answer while a later call runs,
  // so sources and actions keep their place; a preamble before the first call
  // does not count.
  const footerVisible =
    answerVisible ||
    (withToolCalls &&
      nonEmptyTexts(
        message.parts.slice(message.parts.findIndex(isToolCallPart) + 1),
      ).length > 0);
  const reasoningStreaming =
    Boolean(pending) && reasoningText.length > 0 && text === null;
  // Reasoning streaming right now has its own live label, so the wait line steps aside.
  const reasoningLive =
    Boolean(pending) && message.parts.at(-1)?.type === 'reasoning';
  const awaitingLabel = awaitingActivityLabel(message);

  const timing = message.metadata?.timing;
  const diagnostics = message.metadata?.diagnostics;
  const inferenceModel = message.metadata?.inferenceModel;
  const sources = message.metadata?.sources ?? [];
  const responseId = message.metadata?.responseId;
  // One instance for the whole stream, so it keeps its place and toggle state.
  const reasoning =
    reasoningText.length > 0 ? (
      <ReasoningDisclosure
        durationMs={diagnostics?.thinkingMs}
        streaming={reasoningStreaming}
        text={reasoningText}
      />
    ) : null;

  return (
    <Message from="assistant">
      <CitationProvider sources={sources}>
        <MessageContent>
          {/* Once the answer is visible, reasoning collapses to a toggle above it. */}
          {!withToolCalls && answerVisible ? reasoning : null}
          {withToolCalls ? (
            <InterleavedParts
              finishedReasoning={
                !reasoningStreaming && reasoningText.length > 0
                  ? { durationMs: diagnostics?.thinkingMs, text: reasoningText }
                  : undefined
              }
              message={message}
              pending={Boolean(pending)}
              reasoning={reasoning}
            />
          ) : null}
          {!withToolCalls && text ? (
            <div data-testid="answer-text">
              <AnswerMarkdown
                caret="block"
                isAnimating={Boolean(pending)}
              >
                {text}
              </AnswerMarkdown>
            </div>
          ) : null}
          <AssistantMessageStatus
            awaitingLabel={awaitingLabel}
            awaitingModel={awaitingModel && !reasoningLive}
            errorPart={errorPart}
            pending={pending}
            reasoningText={reasoningText}
            statusPart={statusPart}
            text={text}
            toolCallsShown={withToolCalls}
          />
          {/* While still streaming (no answer yet), reasoning streams in BELOW the
            step list so it grows downward instead of displacing the steps. */}
          {withToolCalls || answerVisible ? null : reasoning}
          {errorPart ? (
            <MessageError
              errorPart={errorPart}
              onManageCredentials={onManageCredentials}
              onRetry={onRetry}
              retryPending={retryPending}
            />
          ) : null}
          {/* One quiet row: actions, the sources toggle and the timing, so the
            answer is not followed by a stack of metadata. */}
          {footerVisible ? (
            <MessageFooter
              actions={actions}
              diagnostics={diagnostics}
              inferenceModel={inferenceModel}
              responseId={responseId}
              sources={sources}
              timing={timing}
            />
          ) : null}
        </MessageContent>
      </CitationProvider>
    </Message>
  );
};

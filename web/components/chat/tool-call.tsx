import type { DynamicToolUIPart } from 'ai';

import {
  Check,
  ChevronRight,
  CircleAlert,
  CircleDashed,
  Loader2,
  Search,
  Wrench,
} from 'lucide-react';
import { type ReactNode, useId, useState } from 'react';

import { t } from '@/lib/i18n';
import { searchToolLabel, toolCallIcon } from '@/lib/search-status';
import {
  countLabel,
  tallyToolCalls,
  toolCallInputText,
  toolCallResultBadge,
  toolCallResultText,
  type ToolCallStatus,
  toolCallStatus,
} from '@/lib/tool-calls';

const STATUS_ICONS: Record<ToolCallStatus, ReactNode> = {
  done: <Check className="size-3.5" />,
  error: <CircleAlert className="size-3.5 text-destructive" />,
  incomplete: <CircleDashed className="size-3.5" />,
  running: <Loader2 className="size-3.5 animate-spin" />,
};

const STATUS_LABELS: Record<ToolCallStatus, string> = {
  done: t('toolCall.done'),
  error: t('toolCall.failed'),
  incomplete: t('toolCall.incomplete'),
  running: t('toolCall.running'),
};

export const ToolCall = ({
  className = '',
  part,
  running = false,
}: {
  className?: string;
  part: DynamicToolUIPart;
  running?: boolean;
}) => {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const status = toolCallStatus(part, running);
  const ToolIcon = toolCallIcon(part.toolName);
  const label =
    part.title !== undefined && part.title.length > 0
      ? part.title
      : searchToolLabel(part.toolName);
  const query = toolCallInputText(part.input);
  const result = toolCallResultText(part);
  const badge = toolCallResultBadge(part);

  return (
    <div
      className={`w-fit max-w-full rounded-lg border border-border bg-muted/40 text-sm ${className}`}
      data-state={status}
      data-testid="tool-call"
      data-tool={part.toolName}
    >
      <button
        aria-controls={open ? panelId : undefined}
        aria-expanded={open}
        className="flex w-full min-w-0 items-center gap-2 rounded-lg px-3 py-1.5 text-left text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => {
          setOpen(!open);
        }}
        type="button"
      >
        <ToolIcon
          aria-hidden="true"
          className="size-4 shrink-0"
        />
        <span className="min-w-0 truncate">{label}</span>
        <span
          aria-hidden="true"
          className="inline-flex size-4 shrink-0 items-center justify-center"
        >
          {STATUS_ICONS[status]}
        </span>
        <span className="sr-only">{STATUS_LABELS[status]}</span>
        {badge === null ? null : (
          <span
            className="shrink-0 text-xs tabular-nums text-muted-foreground/80"
            data-testid="tool-call-count"
          >
            {badge}
          </span>
        )}
        <ChevronRight
          aria-hidden="true"
          className={`size-3 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`}
        />
      </button>
      {open ? (
        <dl
          className="flex flex-col gap-1 border-t border-border/60 px-3 py-2 text-xs"
          data-testid="tool-call-panel"
          id={panelId}
        >
          {query === null ? null : (
            <div className="flex gap-2">
              <dt className="shrink-0 text-muted-foreground">
                {t('toolCall.input')}
              </dt>
              <dd className="min-w-0 break-words">{query}</dd>
            </div>
          )}
          {result === null ? null : (
            <div className="flex gap-2">
              <dt className="shrink-0 text-muted-foreground">
                {t('toolCall.result')}
              </dt>
              <dd
                className={`min-w-0 break-words ${status === 'error' ? 'text-destructive' : ''}`}
              >
                {result}
              </dd>
            </div>
          )}
        </dl>
      ) : null}
    </div>
  );
};

// A finished run of calls collapses into one row that expands to the calls.
export const ToolCallGroup = ({
  className = '',
  parts,
}: {
  className?: string;
  parts: readonly DynamicToolUIPart[];
}) => {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const { problems, searches, tools } = tallyToolCalls(parts);
  const GroupIcon = searches > 0 ? Search : Wrench;
  const counts = [
    searches > 0
      ? countLabel(
          searches,
          t('toolCalls.searchOne'),
          t('toolCalls.searchMany'),
        )
      : null,
    tools > 0
      ? countLabel(tools, t('toolCalls.toolOne'), t('toolCalls.toolMany'))
      : null,
  ]
    .filter((label) => label !== null)
    .join(' · ');

  return (
    <div
      className={`flex flex-col gap-2 ${className}`}
      data-testid="tool-call-group"
    >
      <button
        aria-controls={open ? panelId : undefined}
        aria-expanded={open}
        className="flex w-fit max-w-full min-w-0 items-center gap-2 rounded-lg border border-border bg-muted/40 px-3 py-1.5 text-left text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        data-testid="tool-call-summary"
        onClick={() => {
          setOpen(!open);
        }}
        type="button"
      >
        <GroupIcon
          aria-hidden="true"
          className="size-4 shrink-0"
        />
        <span className="min-w-0 truncate">{counts}</span>
        {problems > 0 ? (
          <span
            className="inline-flex shrink-0 items-center gap-1 text-xs tabular-nums text-destructive"
            data-testid="tool-call-problems"
          >
            <CircleAlert
              aria-hidden="true"
              className="size-3.5"
            />
            <span aria-hidden="true">{problems}</span>
            <span className="sr-only">
              {`${problems} ${t('toolCalls.problems')}`}
            </span>
          </span>
        ) : null}
        <ChevronRight
          aria-hidden="true"
          className={`size-3 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`}
        />
      </button>
      {open ? (
        <div
          className="flex flex-col gap-2 border-l border-border/60 pl-3"
          id={panelId}
        >
          {parts.map((part) => (
            <ToolCall
              key={part.toolCallId}
              part={part}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
};

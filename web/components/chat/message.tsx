import type { ReactNode } from 'react';

import { Clock3 } from 'lucide-react';

import type { MyUIMessage } from '@/lib/api-types';

import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { ElapsedTimer } from '@/components/chat/elapsed-timer';
import { SearchStatus } from '@/components/chat/search-status';
import { TypingIndicator } from '@/components/chat/typing-indicator';
import { formatDuration } from '@/lib/duration';
import { t } from '@/lib/i18n';

export type AssistantMessageProps = {
  actions?: ReactNode;
  errorPart?: { code: string; message: string };
  message: MyUIMessage;
  onRetry?: () => void;
  pending?: boolean;
  statusPart?: { label: string; tool?: string };
  streamStartedAt?: null | number;
};

type Timing = NonNullable<MyUIMessage['metadata']>['timing'];

const lastText = (message: MyUIMessage): null | string => {
  const texts = message.parts.filter(
    (p): p is { text: string; type: 'text' } => p.type === 'text',
  );
  const last = texts.at(-1);
  return last ? last.text : null;
};

const MessageTiming = ({ timing }: { timing: Timing }) => {
  if (timing === undefined) {
    return null;
  }

  return (
    <div
      className="mt-1 inline-flex items-center gap-1 text-xs tabular-nums text-muted-foreground/70"
      data-testid="message-timing"
    >
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
    </div>
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
  const text = lastText(message);
  const showChip = Boolean(statusPart) && !text;
  const showDots = Boolean(pending) && !text && !statusPart && !errorPart;
  const timing = message.metadata?.timing;
  const liveTimer =
    typeof streamStartedAt === 'number' ? (
      <ElapsedTimer startedAt={streamStartedAt} />
    ) : null;

  return (
    <Message from="assistant">
      <MessageContent>
        {text ? (
          <div data-testid="answer-text">
            <MessageResponse>{text}</MessageResponse>
          </div>
        ) : null}
        {pending || text === null ? null : <MessageTiming timing={timing} />}
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

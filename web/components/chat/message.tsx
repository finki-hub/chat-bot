import type { ReactNode } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { SearchStatus } from '@/components/chat/search-status';
import { t } from '@/lib/i18n';

export type AssistantMessageProps = {
  actions?: ReactNode;
  errorPart?: { code: string; message: string };
  message: MyUIMessage;
  onRetry?: () => void;
  statusPart?: { label: string; tool?: string };
};

// Last text part only — drops the pre-tool preamble (spec §5.2).
const lastText = (message: MyUIMessage): null | string => {
  const texts = message.parts.filter(
    (p): p is { text: string; type: 'text' } => p.type === 'text',
  );
  const last = texts.at(-1);
  return last ? last.text : null;
};

export const AssistantMessage = ({
  actions,
  errorPart,
  message,
  onRetry,
  statusPart,
}: AssistantMessageProps) => {
  const text = lastText(message);
  const showChip = Boolean(statusPart) && !text;
  const isInterrupted = errorPart?.code === 'interrupted';

  return (
    <Message from="assistant">
      <MessageContent>
        {text ? (
          <div data-testid="answer-text">
            <MessageResponse>{text}</MessageResponse>
          </div>
        ) : null}
        {showChip && statusPart ? (
          <SearchStatus
            label={statusPart.label}
            tool={statusPart.tool}
          />
        ) : null}
        {errorPart ? (
          <div
            className="mt-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm"
            role="alert"
          >
            {isInterrupted ? (
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
        ) : null}
        {actions === undefined || actions === null ? null : (
          <div className="mt-2">{actions}</div>
        )}
      </MessageContent>
    </Message>
  );
};

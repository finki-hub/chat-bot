import { Pencil } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { Button } from '@/components/ui/button';
import { ControlTooltip } from '@/components/ui/icon-controls';
import { Textarea } from '@/components/ui/textarea';
import { t } from '@/lib/i18n';
import { joinText } from '@/lib/message-parts';

// A sent question. With `onEdit` it can be changed and resent, which replaces
// everything after it; the control shows on hover or focus.
export const UserMessage = ({
  className,
  message,
  onEdit,
}: {
  className?: string;
  message: MyUIMessage;
  onEdit?: (messageId: string, text: string) => void;
}) => {
  const text = joinText(message);
  const [draft, setDraft] = useState<null | string>(null);
  const next = draft?.trim() ?? '';
  const canSend = next !== '' && next !== text.trim();
  const editing = draft !== null;
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Opening the editor puts the cursor after the existing text.
  useEffect(() => {
    const input = inputRef.current;

    if (!editing || input === null) {
      return;
    }

    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }, [editing]);

  if (draft !== null && onEdit !== undefined) {
    return (
      <Message
        className={className}
        from="user"
      >
        <form
          className="flex w-full min-w-72 flex-col gap-2"
          data-testid="edit-question"
          onSubmit={(event) => {
            event.preventDefault();
            if (!canSend) {
              return;
            }

            setDraft(null);
            onEdit(message.id, next);
          }}
        >
          <Textarea
            aria-label={t('message.editLabel')}
            className="min-h-20"
            onChange={(event) => {
              setDraft(event.target.value);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                setDraft(null);
              }
            }}
            ref={inputRef}
            value={draft}
          />
          <div className="flex justify-end gap-2">
            <Button
              onClick={() => {
                setDraft(null);
              }}
              size="sm"
              type="button"
              variant="outline"
            >
              {t('common.cancel')}
            </Button>
            <Button
              disabled={!canSend}
              size="sm"
              type="submit"
            >
              {t('message.resend')}
            </Button>
          </div>
        </form>
      </Message>
    );
  }

  return (
    <Message
      className={className}
      from="user"
    >
      {/* The control sits beside the bubble, so it adds no height to the thread. */}
      <div className="flex max-w-full items-start justify-end gap-1">
        {onEdit === undefined ? null : (
          <ControlTooltip label={t('message.edit')}>
            <button
              aria-label={t('message.edit')}
              className="inline-flex size-11 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100 pointer-coarse:opacity-100 sm:pointer-fine:size-8"
              data-testid="edit-question-button"
              onClick={() => {
                setDraft(text);
              }}
              type="button"
            >
              <Pencil
                aria-hidden="true"
                className="size-3.5"
              />
            </button>
          </ControlTooltip>
        )}
        <MessageContent>
          <MessageResponse className="text-balance">{text}</MessageResponse>
        </MessageContent>
      </div>
    </Message>
  );
};

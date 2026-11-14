import { useId, useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import {
  FEEDBACK_COMMENT_MAX,
  FEEDBACK_REASONS,
  type FeedbackReason,
} from '@/lib/api-types';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

export type FeedbackDetails = {
  comment?: string;
  reason?: FeedbackReason;
};

// Asked once after a thumbs-down: why it missed, plus an optional note. Skipping
// keeps the plain dislike that was already recorded.
export const FeedbackReasonDialog = ({
  onOpenChange,
  onSubmit,
  open,
}: {
  onOpenChange: (open: boolean) => void;
  onSubmit: (details: FeedbackDetails) => Promise<boolean>;
  open: boolean;
}) => {
  const [reason, setReason] = useState<FeedbackReason | null>(null);
  const [comment, setComment] = useState('');
  const [sending, setSending] = useState(false);
  const commentId = useId();
  const canSend = !sending && (reason !== null || comment.trim() !== '');

  const send = async (): Promise<void> => {
    setSending(true);
    try {
      const saved = await onSubmit({
        ...(reason !== null && { reason }),
        ...(comment.trim() !== '' && { comment: comment.trim() }),
      });
      if (saved) {
        onOpenChange(false);
      }
    } finally {
      setSending(false);
    }
  };

  return (
    <Dialog
      onOpenChange={(next) => {
        if (!sending) {
          onOpenChange(next);
        }
      }}
      open={open}
    >
      <DialogContent
        aria-describedby={undefined}
        className="sm:max-w-md"
        data-testid="feedback-reason-dialog"
      >
        <DialogHeader>
          <DialogTitle>{t('feedback.title')}</DialogTitle>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (canSend) {
              void send();
            }
          }}
        >
          <fieldset className="flex flex-wrap gap-2">
            <legend className="sr-only">{t('feedback.reasons')}</legend>
            {FEEDBACK_REASONS.map((option) => (
              <button
                aria-pressed={reason === option}
                className={cn(
                  'min-h-11 rounded-full border border-border px-3 text-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-fine:min-h-8',
                  reason === option &&
                    'border-primary bg-primary/10 text-foreground',
                )}
                key={option}
                onClick={() => {
                  setReason(reason === option ? null : option);
                }}
                type="button"
              >
                {t(`feedback.reason.${option}`)}
              </button>
            ))}
          </fieldset>
          <div className="flex flex-col gap-1.5">
            <label
              className="text-sm text-muted-foreground"
              htmlFor={commentId}
            >
              {t('feedback.comment')}
            </label>
            <Textarea
              id={commentId}
              maxLength={FEEDBACK_COMMENT_MAX}
              onChange={(event) => {
                setComment(event.target.value);
              }}
              rows={3}
              value={comment}
            />
          </div>
          <DialogFooter>
            <Button
              disabled={sending}
              onClick={() => {
                onOpenChange(false);
              }}
              type="button"
              variant="outline"
            >
              {t('feedback.skip')}
            </Button>
            <Button
              aria-busy={sending || undefined}
              disabled={!canSend}
              type="submit"
            >
              {t('feedback.send')}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

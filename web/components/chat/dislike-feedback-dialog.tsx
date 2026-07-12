import { useState } from 'react';

import type { DislikeReasonCategory } from '@/lib/api-types';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { t } from '@/lib/i18n';

export type DislikeFeedback = {
  readonly category: DislikeReasonCategory;
  readonly detail?: string;
};

type DislikeFeedbackDialogProps = {
  readonly onOpenChange: (open: boolean) => void;
  readonly onSubmit: (feedback: DislikeFeedback) => Promise<boolean>;
  readonly open: boolean;
};

const categories: readonly DislikeReasonCategory[] = [
  'incorrect',
  'incomplete',
  'off_topic',
  'outdated',
  'other',
];

export const DislikeFeedbackDialog = ({
  onOpenChange,
  onSubmit,
  open,
}: DislikeFeedbackDialogProps) => {
  const [category, setCategory] = useState<DislikeReasonCategory | null>(null);
  const [detail, setDetail] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const submit = async (): Promise<void> => {
    if (category === null) {
      return;
    }
    setSubmitting(true);
    try {
      const trimmed = detail.trim();
      const accepted = await onSubmit({
        category,
        ...(trimmed && { detail: trimmed }),
      });
      if (accepted) {
        setCategory(null);
        setDetail('');
        onOpenChange(false);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog
      onOpenChange={onOpenChange}
      open={open}
    >
      <DialogContent
        aria-describedby={undefined}
        className="sm:max-w-md"
      >
        <DialogHeader>
          <DialogTitle>{t('feedback.dislikeTitle')}</DialogTitle>
        </DialogHeader>
        <fieldset className="space-y-2">
          <legend className="text-sm text-muted-foreground">
            {t('feedback.dislikePrompt')}
          </legend>
          {categories.map((value) => (
            <label
              className="flex min-h-11 items-center gap-3 rounded-lg border px-3 py-2"
              key={value}
            >
              <input
                aria-label={t(`feedback.reason.${value}`)}
                checked={category === value}
                name="dislike-reason"
                onChange={() => {
                  setCategory(value);
                }}
                type="radio"
              />
              <span>{t(`feedback.reason.${value}`)}</span>
            </label>
          ))}
        </fieldset>
        <label className="space-y-2 text-sm">
          <span>{t('feedback.detail')}</span>
          <textarea
            aria-label={t('feedback.detail')}
            className="min-h-24 w-full resize-y rounded-md border border-input bg-background p-3 outline-none focus-visible:ring-2 focus-visible:ring-ring"
            maxLength={500}
            onChange={(event) => {
              setDetail(event.target.value);
            }}
            value={detail}
          />
          <span className="block text-right text-xs text-muted-foreground">
            {detail.length}/500
          </span>
        </label>
        <DialogFooter>
          <Button
            onClick={() => {
              onOpenChange(false);
            }}
            type="button"
            variant="outline"
          >
            {t('common.cancel')}
          </Button>
          <Button
            disabled={category === null || submitting}
            onClick={async () => {
              await submit();
            }}
            type="button"
          >
            {t('feedback.submit')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

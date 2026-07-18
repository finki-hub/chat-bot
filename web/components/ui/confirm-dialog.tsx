import { type ReactNode, useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Spinner } from '@/components/ui/spinner';
import type { MaybeAsyncAction } from '@/lib/action-result';
import { fireAndForget } from '@/lib/async';
import { t } from '@/lib/i18n';

export type ConfirmDialogProps = {
  cancelLabel?: string;
  confirmLabel: string;
  description: ReactNode;
  destructive?: boolean;
  errorMessage?: string;
  onConfirm: MaybeAsyncAction;
  onOpenChange: (open: boolean) => void;
  open: boolean;
  title: string;
};

export const ConfirmDialog = ({
  cancelLabel,
  confirmLabel,
  description,
  destructive = false,
  errorMessage,
  onConfirm,
  onOpenChange,
  open,
  title,
}: ConfirmDialogProps) => {
  const [failed, setFailed] = useState(false);
  const [pending, setPending] = useState(false);

  const confirm = async (): Promise<void> => {
    setFailed(false);
    setPending(true);
    try {
      const confirmed = await onConfirm();
      if (confirmed === false) {
        setFailed(true);
        return;
      }
      onOpenChange(false);
    } finally {
      setPending(false);
    }
  };

  return (
    <Dialog
      onOpenChange={(nextOpen) => {
        if (!pending) {
          if (!nextOpen) {
            setFailed(false);
          }
          onOpenChange(nextOpen);
        }
      }}
      open={open}
    >
      <DialogContent
        className="sm:max-w-sm"
        showCloseButton={!pending}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        {failed && errorMessage !== undefined ? (
          <p
            className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
            role="alert"
          >
            {errorMessage}
          </p>
        ) : null}
        <DialogFooter>
          <Button
            disabled={pending}
            onClick={() => {
              setFailed(false);
              onOpenChange(false);
            }}
            type="button"
            variant="outline"
          >
            {cancelLabel ?? t('common.cancel')}
          </Button>
          <Button
            aria-busy={pending || undefined}
            data-testid="confirm-action"
            disabled={pending}
            onClick={() => {
              if (!pending) {
                fireAndForget(confirm());
              }
            }}
            type="button"
            variant={destructive ? 'destructive' : 'default'}
          >
            {pending ? <Spinner aria-hidden="true" /> : null}
            {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

'use client';

import type { ReactNode } from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { t } from '@/lib/i18n';

export type ConfirmDialogProps = {
  cancelLabel?: string;
  confirmLabel: string;
  description: ReactNode;
  destructive?: boolean;
  onConfirm: () => void;
  onOpenChange: (open: boolean) => void;
  open: boolean;
  title: string;
};

export const ConfirmDialog = ({
  cancelLabel,
  confirmLabel,
  description,
  destructive = false,
  onConfirm,
  onOpenChange,
  open,
  title,
}: ConfirmDialogProps) => (
  <Dialog
    onOpenChange={onOpenChange}
    open={open}
  >
    <DialogContent className="sm:max-w-sm">
      <DialogHeader>
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
      </DialogHeader>
      <DialogFooter>
        <Button
          onClick={() => {
            onOpenChange(false);
          }}
          type="button"
          variant="outline"
        >
          {cancelLabel ?? t('common.cancel')}
        </Button>
        <Button
          data-testid="confirm-action"
          onClick={() => {
            onConfirm();
            onOpenChange(false);
          }}
          type="button"
          variant={destructive ? 'destructive' : 'default'}
        >
          {confirmLabel}
        </Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
);

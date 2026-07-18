import type { MaybeAsyncAction } from '@/lib/action-result';

import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { t } from '@/lib/i18n';

type SidebarClearDialogProps = {
  readonly onConfirm: MaybeAsyncAction;
  readonly onOpenChangeAction: (open: boolean) => void;
  readonly open: boolean;
};

export const SidebarClearDialog = ({
  onConfirm,
  onOpenChangeAction,
  open,
}: SidebarClearDialogProps) => (
  <ConfirmDialog
    confirmLabel={t('conversation.delete')}
    description={t('conversation.deleteAllDescription')}
    destructive
    errorMessage={t('conversation.deleteAllError')}
    onConfirm={onConfirm}
    onOpenChange={onOpenChangeAction}
    open={open}
    title={t('conversation.deleteAllTitle')}
  />
);

import type { MaybeAsyncAction } from '@/lib/action-result';
import type { ChatCredentialProvider } from '@/lib/api-types';

import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { t } from '@/lib/i18n';

type CredentialDeleteDialogProps = {
  readonly onConfirm: MaybeAsyncAction<[provider: ChatCredentialProvider]>;
  readonly onOpenChange: (open: boolean) => void;
  readonly provider: ChatCredentialProvider | null;
};

export const CredentialDeleteDialog = ({
  onConfirm,
  onOpenChange,
  provider,
}: CredentialDeleteDialogProps) => (
  <ConfirmDialog
    confirmLabel={t('common.delete')}
    description={t('settings.deleteCredentialDescription')}
    destructive
    errorMessage={t('settings.credentialDeleteError')}
    onConfirm={() => (provider === null ? undefined : onConfirm(provider))}
    onOpenChange={onOpenChange}
    open={provider !== null}
    title={t('settings.deleteCredential')}
  />
);

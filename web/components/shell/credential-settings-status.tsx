import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { t } from '@/lib/i18n';

type CredentialSettingsStatusProps = {
  readonly loadError: boolean;
  readonly loading: boolean;
  readonly onRetryAction: () => void;
};

export const CredentialSettingsStatus = ({
  loadError,
  loading,
  onRetryAction,
}: CredentialSettingsStatusProps) => {
  if (loading) {
    return (
      <output className="flex items-center gap-2 rounded-xl border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        <Spinner aria-hidden="true" />
        {t('settings.credentialsLoading')}
      </output>
    );
  }

  if (!loadError) {
    return null;
  }

  return (
    <div
      className="flex items-center justify-between gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive"
      role="alert"
    >
      <p>{t('settings.credentialsError')}</p>
      <Button
        className="pointer-coarse:min-h-11"
        onClick={onRetryAction}
        size="sm"
        type="button"
        variant="outline"
      >
        {t('error.retry')}
      </Button>
    </div>
  );
};

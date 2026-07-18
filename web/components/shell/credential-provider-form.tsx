import type { SyntheticEvent } from 'react';

import { KeyRound, Trash2 } from 'lucide-react';

import type { ProviderForm } from '@/components/shell/credential-settings-data';
import type { ChatCredentialPublic } from '@/lib/api-types';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Spinner } from '@/components/ui/spinner';
import { t } from '@/lib/i18n';

type CredentialProviderFormProps = {
  readonly busy: boolean;
  readonly credential?: ChatCredentialPublic;
  readonly form: ProviderForm;
  readonly label: string;
  readonly onDelete: () => void;
  readonly onFieldChange: (field: keyof ProviderForm, value: string) => void;
  readonly onSubmit: (event: SyntheticEvent<HTMLFormElement>) => void;
};

const actionLabel = (busy: boolean): string =>
  busy ? t('composer.modelsLoading') : t('common.save');

export const CredentialProviderForm = ({
  busy,
  credential,
  form,
  label,
  onDelete,
  onFieldChange,
  onSubmit,
}: CredentialProviderFormProps) => (
  <form
    className="rounded-xl border border-border bg-card p-4"
    onSubmit={onSubmit}
  >
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <KeyRound
            aria-hidden="true"
            className="text-muted-foreground"
          />
          <div>
            <h3 className="text-sm font-semibold">{label}</h3>
            <p className="text-xs text-muted-foreground">
              {credential === undefined
                ? t('settings.noCredential')
                : t('settings.savedCredential')}
            </p>
          </div>
        </div>
        {credential === undefined ? null : (
          <Button
            aria-busy={busy || undefined}
            className="pointer-coarse:min-h-11"
            disabled={busy}
            onClick={onDelete}
            size="sm"
            type="button"
            variant="destructive"
          >
            <Trash2 data-icon="inline-start" />
            {t('common.delete')}
          </Button>
        )}
      </div>
      <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
        <Input
          aria-label={`${label} API key`}
          autoComplete="off"
          onChange={(event) => {
            onFieldChange('apiKey', event.target.value);
          }}
          placeholder={t('settings.keyPlaceholder')}
          type="password"
          value={form.apiKey}
        />
        <Input
          aria-label={`${label} base URL`}
          onChange={(event) => {
            onFieldChange('baseUrl', event.target.value);
          }}
          placeholder={credential?.base_url ?? t('settings.baseUrl')}
          type="url"
          value={form.baseUrl}
        />
        <Button
          aria-busy={busy || undefined}
          className="pointer-coarse:min-h-11"
          disabled={busy || form.apiKey.trim().length === 0}
          type="submit"
        >
          {busy ? <Spinner aria-hidden="true" /> : null}
          {actionLabel(busy)}
        </Button>
      </div>
    </div>
  </form>
);

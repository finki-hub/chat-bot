import { ExternalLink, KeyRound, Trash2 } from 'lucide-react';

import type { ProviderForm } from '@/components/shell/credential-settings-data';
import type { ChatCredentialPublic } from '@/lib/api-types';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { t } from '@/lib/i18n';

type CredentialProviderFormProps = {
  readonly busy: boolean;
  readonly credential?: ChatCredentialPublic;
  readonly form: ProviderForm;
  readonly keyUrl: string;
  readonly label: string;
  readonly onDelete: () => void;
  readonly onFieldChange: (field: keyof ProviderForm, value: string) => void;
};

export const CredentialProviderForm = ({
  busy,
  credential,
  form,
  keyUrl,
  label,
  onDelete,
  onFieldChange,
}: CredentialProviderFormProps) => (
  <section className="rounded-xl border border-border bg-card p-4">
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 gap-y-1">
        <div className="flex items-center gap-2">
          <KeyRound
            aria-hidden="true"
            className="text-muted-foreground"
          />
          <h3 className="text-sm font-semibold">{label}</h3>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button
            asChild
            className="pointer-coarse:min-h-11 pointer-coarse:min-w-11"
            size="sm"
            variant="ghost"
          >
            <a
              aria-label={`${t('settings.getApiKey')}: ${label}`}
              href={keyUrl}
              rel="noreferrer"
              target="_blank"
            >
              <ExternalLink data-icon="inline-start" />
              <span className="hidden sm:inline">
                {t('settings.getApiKey')}
              </span>
            </a>
          </Button>
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
        <p className="col-span-2 text-xs text-muted-foreground sm:col-span-1 sm:pl-8">
          {credential === undefined
            ? t('settings.noCredential')
            : t('settings.savedCredential')}
        </p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <Input
          aria-label={`${label} API key`}
          autoComplete="off"
          disabled={busy}
          onChange={(event) => {
            onFieldChange('apiKey', event.target.value);
          }}
          placeholder={t('settings.keyPlaceholder')}
          type="password"
          value={form.apiKey}
        />
        <Input
          aria-label={`${label} base URL`}
          disabled={busy}
          onChange={(event) => {
            onFieldChange('baseUrl', event.target.value);
          }}
          placeholder={credential?.base_url ?? t('settings.baseUrl')}
          type="url"
          value={form.baseUrl}
        />
      </div>
    </div>
  </section>
);

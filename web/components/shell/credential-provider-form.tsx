import {
  Circle,
  CircleCheck,
  ExternalLink,
  KeyRound,
  Trash2,
} from 'lucide-react';
import { useId } from 'react';

import type { ProviderForm } from '@/components/shell/credential-settings-data';
import type { CredentialSaveFailure } from '@/components/shell/credential-settings-save';
import type { ChatCredentialPublic } from '@/lib/api-types';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { t } from '@/lib/i18n';

type CredentialProviderFormProps = {
  readonly busy: boolean;
  readonly credential?: ChatCredentialPublic;
  readonly failure?: CredentialSaveFailure;
  readonly form: ProviderForm;
  readonly keyUrl: string;
  readonly label: string;
  readonly onDelete: () => void;
  readonly onFieldChange: (field: keyof ProviderForm, value: string) => void;
};

export const CredentialProviderForm = ({
  busy,
  credential,
  failure,
  form,
  keyUrl,
  label,
  onDelete,
  onFieldChange,
}: CredentialProviderFormProps) => {
  const fieldId = useId();
  const apiKeyId = `${fieldId}-api-key`;
  const apiKeyDescriptionId = `${fieldId}-api-key-description`;
  const apiKeyErrorId = `${fieldId}-api-key-error`;
  const baseUrlId = `${fieldId}-base-url`;
  const baseUrlDescriptionId = `${fieldId}-base-url-description`;
  const baseUrlErrorId = `${fieldId}-base-url-error`;
  const hasCredential = credential !== undefined;
  /* eslint-disable @typescript-eslint/prefer-nullish-coalescing -- legacy empty URLs should use the example. */
  const baseUrlPlaceholder =
    credential?.base_url || t('settings.baseUrlPlaceholder');
  /* eslint-enable @typescript-eslint/prefer-nullish-coalescing -- end legacy empty URL exception. */

  return (
    <section className="rounded-xl border border-border bg-card p-4">
      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <div className="flex min-w-0 items-center gap-2">
              <KeyRound
                aria-hidden="true"
                className="size-4 shrink-0 text-muted-foreground"
              />
              <h3 className="whitespace-nowrap text-sm font-semibold">
                {label}
              </h3>
            </div>
            <output
              className={
                hasCredential
                  ? 'inline-flex items-center gap-1 rounded-md border border-success/25 bg-success/10 px-2 py-1 text-xs font-medium text-success'
                  : 'inline-flex items-center gap-1 rounded-md border border-border bg-muted px-2 py-1 text-xs font-medium text-muted-foreground'
              }
            >
              {hasCredential ? (
                <CircleCheck
                  aria-hidden="true"
                  className="size-3.5"
                />
              ) : (
                <Circle
                  aria-hidden="true"
                  className="size-3.5"
                />
              )}
              {hasCredential
                ? t('settings.savedCredential')
                : t('settings.noCredential')}
            </output>
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
            {hasCredential ? (
              <Button
                aria-busy={busy || undefined}
                className="text-destructive hover:bg-destructive/10 hover:text-destructive pointer-coarse:min-h-11"
                disabled={busy}
                onClick={onDelete}
                size="sm"
                type="button"
                variant="ghost"
              >
                <Trash2 data-icon="inline-start" />
                {t('common.delete')}
              </Button>
            ) : null}
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 sm:items-start">
          <div className="grid gap-1.5">
            <label
              className="text-xs font-medium"
              htmlFor={apiKeyId}
            >
              <span className="sr-only">{label}</span> {t('settings.apiKey')}
            </label>
            <Input
              aria-describedby={
                failure?.field === 'apiKey'
                  ? `${apiKeyDescriptionId} ${apiKeyErrorId}`
                  : apiKeyDescriptionId
              }
              autoComplete="off"
              disabled={busy}
              id={apiKeyId}
              onChange={(event) => {
                onFieldChange('apiKey', event.target.value);
              }}
              placeholder={t('settings.keyPlaceholder')}
              type="password"
              value={form.apiKey}
            />
            <p
              className="text-pretty text-xs leading-relaxed text-muted-foreground"
              id={apiKeyDescriptionId}
            >
              {hasCredential
                ? t('settings.replaceCredential')
                : t('settings.optionalCredential')}
            </p>
            {failure?.field === 'apiKey' ? (
              <p
                className="text-pretty text-xs leading-relaxed text-destructive"
                id={apiKeyErrorId}
              >
                {t('settings.credentialSaveError')}
              </p>
            ) : null}
          </div>
          <div className="grid gap-1.5">
            <label
              className="text-xs font-medium"
              htmlFor={baseUrlId}
            >
              <span className="sr-only">{label}</span> {t('settings.baseUrl')}
            </label>
            <Input
              aria-describedby={
                failure?.field === 'baseUrl'
                  ? `${baseUrlDescriptionId} ${baseUrlErrorId}`
                  : baseUrlDescriptionId
              }
              aria-invalid={failure?.field === 'baseUrl' || undefined}
              disabled={busy}
              id={baseUrlId}
              onChange={(event) => {
                onFieldChange('baseUrl', event.target.value);
              }}
              placeholder={baseUrlPlaceholder}
              type="url"
              value={form.baseUrl}
            />
            <p
              className="text-pretty text-xs leading-relaxed text-muted-foreground"
              id={baseUrlDescriptionId}
            >
              {t('settings.baseUrlHint')}
            </p>
            {failure?.field === 'baseUrl' ? (
              <p
                className="text-pretty text-xs leading-relaxed text-destructive"
                id={baseUrlErrorId}
              >
                {t('settings.credentialBaseUrlError')}
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
};

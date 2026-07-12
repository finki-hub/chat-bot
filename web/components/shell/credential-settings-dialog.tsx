'use client';

import { useQueryClient } from '@tanstack/react-query';
import { type SyntheticEvent, useEffect, useState } from 'react';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import { CredentialProviderForm } from '@/components/shell/credential-provider-form';
import {
  CredentialBaseUrlRejectedError,
  deleteCredential,
  saveCredential,
} from '@/components/shell/credential-settings-client';
import {
  credentialsByProvider,
  EMPTY_FORMS,
  type ProviderConfig,
  type ProviderForm,
  PROVIDERS,
} from '@/components/shell/credential-settings-data';
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
import { t } from '@/lib/i18n';
import { CREDENTIALS_QUERY_KEY, useCredentials } from '@/lib/use-credentials';

type CredentialSettingsDialogProps = {
  readonly onOpenChange: (open: boolean) => void;
  readonly open: boolean;
};

const CREDENTIALS_ERROR_KEY = 'settings.credentialsError' satisfies Parameters<
  typeof t
>[0];

const credentialsError = (): string => t(CREDENTIALS_ERROR_KEY);

const providerList: readonly ProviderConfig[] = PROVIDERS;

type FormState = Record<ChatCredentialProvider, ProviderForm>;

export const CredentialSettingsDialog = ({
  onOpenChange,
  open,
}: CredentialSettingsDialogProps) => {
  const queryClient = useQueryClient();
  const {
    credentials,
    isError: credentialsLoadError,
    isLoading: loading,
    refetch,
  } = useCredentials();
  const [forms, setForms] = useState<FormState>(EMPTY_FORMS);
  const [busyProvider, setBusyProvider] =
    useState<ChatCredentialProvider | null>(null);
  const [error, setError] = useState<null | string>(null);
  useEffect(() => {
    if (!open) {
      return;
    }
    const loadedForms: FormState = { ...EMPTY_FORMS };
    for (const credential of credentials) {
      loadedForms[credential.provider] = {
        apiKey: '',
        baseUrl: credential.base_url ?? '',
      };
    }
    setForms(loadedForms);
  }, [credentials, open]);
  const saved = credentialsByProvider(credentials);

  const updateForm = (
    provider: ChatCredentialProvider,
    field: keyof ProviderForm,
    value: string,
  ) => {
    setForms((current) => ({
      ...current,
      [provider]: { ...current[provider], [field]: value },
    }));
  };
  const saveProvider = async (
    event: SyntheticEvent<HTMLFormElement>,
    provider: ChatCredentialProvider,
  ) => {
    event.preventDefault();
    const form = forms[provider];
    const apiKey = form.apiKey.trim();
    if (apiKey.length === 0) {
      return;
    }
    setBusyProvider(provider);
    setError(null);
    try {
      const credential = await saveCredential({
        apiKey,
        baseUrl: form.baseUrl.trim(),
        provider,
      });
      if (credential === null) {
        setError(t('settings.credentialSaveError'));
        return;
      }
      queryClient.setQueryData<null | readonly ChatCredentialPublic[]>(
        CREDENTIALS_QUERY_KEY,
        (current) => [
          ...(current ?? []).filter((item) => item.provider !== provider),
          credential,
        ],
      );
      await queryClient.invalidateQueries({ queryKey: CREDENTIALS_QUERY_KEY });
      setForms((current) => ({
        ...current,
        [provider]: { apiKey: '', baseUrl: credential.base_url ?? '' },
      }));
    } catch (error_) {
      if (error_ instanceof CredentialBaseUrlRejectedError) {
        setError(t('settings.credentialBaseUrlError'));
      } else if (error_ instanceof TypeError) {
        setError(t('settings.credentialSaveError'));
      } else {
        throw error_;
      }
    } finally {
      setBusyProvider(null);
    }
  };

  const deleteProvider = async (provider: ChatCredentialProvider) => {
    setBusyProvider(provider);
    setError(null);
    try {
      const deleted = await deleteCredential(provider);
      if (!deleted) {
        setError(t('settings.credentialDeleteError'));
        return;
      }
      queryClient.setQueryData<null | readonly ChatCredentialPublic[]>(
        CREDENTIALS_QUERY_KEY,
        (current) =>
          (current ?? []).filter(
            (credential) => credential.provider !== provider,
          ),
      );
      await queryClient.invalidateQueries({ queryKey: CREDENTIALS_QUERY_KEY });
      setForms((current) => ({
        ...current,
        [provider]: EMPTY_FORMS[provider],
      }));
    } catch (error_) {
      if (!(error_ instanceof TypeError)) {
        throw error_;
      }
      setError(t('settings.credentialDeleteError'));
    } finally {
      setBusyProvider(null);
    }
  };

  return (
    <Dialog
      onOpenChange={onOpenChange}
      open={open}
    >
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t('settings.credentialsTitle')}</DialogTitle>
          <DialogDescription>
            {t('settings.credentialsDescription')}
          </DialogDescription>
        </DialogHeader>
        {loading ? (
          <div className="flex items-center gap-2 rounded-xl border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
            <Spinner aria-hidden="true" />
            {t('composer.modelsLoading')}
          </div>
        ) : null}
        {!loading && credentialsLoadError ? (
          <div
            className="flex items-center justify-between gap-3 rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive"
            role="alert"
          >
            <p>{credentialsError()}</p>
            <Button
              onClick={() => {
                void refetch();
              }}
              size="sm"
              type="button"
              variant="outline"
            >
              {t('error.retry')}
            </Button>
          </div>
        ) : null}
        {!loading && !credentialsLoadError ? (
          <div className="flex flex-col gap-3">
            {providerList.map(({ labelKey, provider }) => {
              const credential = saved[provider];
              const busy = busyProvider === provider;
              return (
                <CredentialProviderForm
                  busy={busy}
                  credential={credential}
                  form={forms[provider]}
                  key={provider}
                  label={t(labelKey)}
                  onDelete={() => {
                    void deleteProvider(provider);
                  }}
                  onFieldChange={(field, value) => {
                    updateForm(provider, field, value);
                  }}
                  onSubmit={(event) => {
                    void saveProvider(event, provider);
                  }}
                />
              );
            })}
          </div>
        ) : null}
        {error === null ? null : (
          <p className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

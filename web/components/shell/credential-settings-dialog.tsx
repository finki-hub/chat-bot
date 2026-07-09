'use client';

import { type SyntheticEvent, useEffect, useState } from 'react';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import { CredentialProviderForm } from '@/components/shell/credential-provider-form';
import {
  deleteCredential,
  loadCredentials,
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

type CredentialSettingsDialogProps = {
  readonly onOpenChange: (open: boolean) => void;
  readonly open: boolean;
};

const CREDENTIALS_ERROR_KEY = 'settings.credentialsError' satisfies Parameters<
  typeof t
>[0];

const credentialsError = (): string => t(CREDENTIALS_ERROR_KEY);

const noop = () => {};

const providerList: readonly ProviderConfig[] = PROVIDERS;

type FormState = Record<ChatCredentialProvider, ProviderForm>;

export const CredentialSettingsDialog = ({
  onOpenChange,
  open,
}: CredentialSettingsDialogProps) => {
  const [credentials, setCredentials] = useState<
    readonly ChatCredentialPublic[]
  >([]);
  const [forms, setForms] = useState<FormState>(EMPTY_FORMS);
  const [loading, setLoading] = useState(false);
  const [busyProvider, setBusyProvider] =
    useState<ChatCredentialProvider | null>(null);
  const [error, setError] = useState<null | string>(null);

  useEffect(() => {
    if (!open) {
      return noop;
    }

    const controller = new AbortController();
    const loadSavedCredentials = async (): Promise<void> => {
      setLoading(true);
      setError(null);
      try {
        const loadedCredentials = await loadCredentials(controller.signal);
        if (loadedCredentials === null) {
          setError(credentialsError());
          return;
        }
        setCredentials(loadedCredentials);
      } catch (error_) {
        if (error_ instanceof DOMException && error_.name === 'AbortError') {
          return;
        }
        if (error_ instanceof TypeError) {
          setError(credentialsError());
          return;
        }
        throw error_;
      } finally {
        setLoading(false);
      }
    };

    void loadSavedCredentials();
    return () => {
      controller.abort();
    };
  }, [open]);

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
        setError(credentialsError());
        return;
      }
      setCredentials((current) => [
        ...current.filter((item) => item.provider !== provider),
        credential,
      ]);
      setForms((current) => ({
        ...current,
        [provider]: { apiKey: '', baseUrl: credential.base_url ?? '' },
      }));
    } catch (error_) {
      if (!(error_ instanceof TypeError)) {
        throw error_;
      }
      setError(credentialsError());
    } finally {
      setBusyProvider(null);
    }
  };

  const deleteProvider = async (
    provider: ChatCredentialProvider,
  ): Promise<void> => {
    setBusyProvider(provider);
    setError(null);
    try {
      const deleted = await deleteCredential(provider);
      if (!deleted) {
        setError(credentialsError());
        return;
      }
      setCredentials((current) =>
        current.filter((credential) => credential.provider !== provider),
      );
    } catch (error_) {
      if (!(error_ instanceof TypeError)) {
        throw error_;
      }
      setError(credentialsError());
    } finally {
      setBusyProvider(null);
    }
  };

  return (
    <Dialog
      onOpenChange={onOpenChange}
      open={open}
    >
      <DialogContent className="sm:max-w-2xl">
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
        ) : (
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
        )}
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

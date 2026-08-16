'use client';

import { useQueryClient } from '@tanstack/react-query';
import { type SyntheticEvent, useEffect, useRef, useState } from 'react';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import { CredentialDeleteDialog } from '@/components/shell/credential-delete-dialog';
import { CredentialProviderForm } from '@/components/shell/credential-provider-form';
import { deleteCredential } from '@/components/shell/credential-settings-client';
import {
  credentialsByProvider,
  EMPTY_FORMS,
  type ProviderConfig,
  type ProviderForm,
  PROVIDERS,
} from '@/components/shell/credential-settings-data';
import {
  type CredentialSaveFailure,
  saveEnteredCredentials,
} from '@/components/shell/credential-settings-save';
import { CredentialSettingsStatus } from '@/components/shell/credential-settings-status';
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
import { useCredentials } from '@/lib/use-credentials';
import { useModels } from '@/lib/use-models';

type CredentialSettingsDialogProps = {
  readonly onOpenChangeAction: (open: boolean) => void;
  readonly open: boolean;
};

const providerList: readonly ProviderConfig[] = PROVIDERS;

type CredentialProviderListProps = {
  readonly busyProvider: ChatCredentialProvider | null;
  readonly forms: FormState;
  readonly onDelete: (provider: ChatCredentialProvider) => void;
  readonly onFieldChange: (
    provider: ChatCredentialProvider,
    field: keyof ProviderForm,
    value: string,
  ) => void;
  readonly saved: SavedCredentials;
  readonly saving: boolean;
};

type FormState = Record<ChatCredentialProvider, ProviderForm>;

type SavedCredentials = Partial<
  Record<ChatCredentialProvider, ChatCredentialPublic>
>;

const saveFailureMessage = (
  failure: CredentialSaveFailure | null,
): null | string => {
  if (failure === null) {
    return null;
  }
  return t(
    failure === 'base-url'
      ? 'settings.credentialBaseUrlError'
      : 'settings.credentialSaveError',
  );
};

const formsForCredentials = (
  credentials: readonly ChatCredentialPublic[],
): FormState => {
  const forms: FormState = { ...EMPTY_FORMS };
  for (const credential of credentials) {
    forms[credential.provider] = {
      apiKey: '',
      baseUrl: credential.base_url ?? '',
    };
  }
  return forms;
};

const formsWithSavedCredentials = (
  current: FormState,
  credentials: readonly ChatCredentialPublic[],
): FormState => {
  let next = current;
  for (const credential of credentials) {
    next = {
      ...next,
      [credential.provider]: {
        apiKey: '',
        baseUrl: credential.base_url ?? '',
      },
    };
  }
  return next;
};

const formsWithPendingDrafts = (
  current: FormState,
  credentials: readonly ChatCredentialPublic[],
): FormState => {
  let next = formsForCredentials(credentials);
  for (const { provider } of providerList) {
    if (current[provider].apiKey.trim().length > 0) {
      next = { ...next, [provider]: current[provider] };
    }
  }
  return next;
};

const CredentialProviderList = ({
  busyProvider,
  forms,
  onDelete,
  onFieldChange,
  saved,
  saving,
}: CredentialProviderListProps) => (
  <div className="flex flex-col gap-3">
    {providerList.map(({ labelKey, provider }) => (
      <CredentialProviderForm
        busy={saving || busyProvider === provider}
        credential={saved[provider]}
        form={forms[provider]}
        key={provider}
        label={t(labelKey)}
        onDelete={() => {
          onDelete(provider);
        }}
        onFieldChange={(field, value) => {
          onFieldChange(provider, field, value);
        }}
      />
    ))}
  </div>
);

export const CredentialSettingsDialog = ({
  onOpenChangeAction,
  open,
}: CredentialSettingsDialogProps) => {
  const queryClient = useQueryClient();
  const {
    credentials,
    isError: credentialsLoadError,
    isLoading: loading,
    queryKey,
    refetch,
    sessionKey,
  } = useCredentials();
  const { refetch: refetchModels } = useModels();
  const [forms, setForms] = useState<FormState>(EMPTY_FORMS);
  const [busyProvider, setBusyProvider] =
    useState<ChatCredentialProvider | null>(null);
  const [credentialToDelete, setCredentialToDelete] =
    useState<ChatCredentialProvider | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<null | string>(null);
  const sessionKeyRef = useRef(sessionKey);
  sessionKeyRef.current = sessionKey;
  useEffect(() => {
    setBusyProvider(null);
    setCredentialToDelete(null);
    setError(null);
    setForms(EMPTY_FORMS);
    setSaving(false);
  }, [sessionKey]);
  useEffect(() => {
    if (!open) {
      setCredentialToDelete(null);
      setForms(EMPTY_FORMS);
      return;
    }
    setForms((current) => formsWithPendingDrafts(current, credentials));
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
  const saveProviders = async (event: SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (sessionKey === null) {
      return;
    }
    const submittingSessionKey = sessionKey;
    setSaving(true);
    setError(null);
    try {
      const {
        credentials: savedCredentials,
        failure,
        unexpectedError,
      } = await saveEnteredCredentials(forms);
      if (sessionKeyRef.current !== submittingSessionKey) {
        return;
      }
      if (savedCredentials.length > 0) {
        const savedProviders = new Set(
          savedCredentials.map((credential) => credential.provider),
        );
        queryClient.setQueryData<null | readonly ChatCredentialPublic[]>(
          queryKey,
          (current) => [
            ...(current ?? []).filter(
              (credential) => !savedProviders.has(credential.provider),
            ),
            ...savedCredentials,
          ],
        );
        await queryClient.invalidateQueries({
          exact: true,
          queryKey,
        });
        if (sessionKeyRef.current !== submittingSessionKey) {
          return;
        }
        await refetchModels();
        setForms((current) =>
          formsWithSavedCredentials(current, savedCredentials),
        );
      }
      setError(saveFailureMessage(failure));
      if (unexpectedError !== null) {
        reportError(unexpectedError.reason);
      }
    } catch (error_) {
      if (error_ instanceof TypeError) {
        setError(t('settings.credentialSaveError'));
      } else {
        throw error_;
      }
    } finally {
      if (sessionKeyRef.current === submittingSessionKey) {
        setSaving(false);
      }
    }
  };

  const deleteProvider = async (
    provider: ChatCredentialProvider,
  ): Promise<boolean> => {
    if (sessionKey === null) {
      return false;
    }
    const deletingSessionKey = sessionKey;
    setBusyProvider(provider);
    setError(null);
    try {
      const deleted = await deleteCredential(provider);
      if (!deleted) {
        setError(t('settings.credentialDeleteError'));
        return false;
      }
      if (sessionKeyRef.current !== deletingSessionKey) {
        return false;
      }
      queryClient.setQueryData<null | readonly ChatCredentialPublic[]>(
        queryKey,
        (current) =>
          (current ?? []).filter(
            (credential) => credential.provider !== provider,
          ),
      );
      await queryClient.invalidateQueries({ exact: true, queryKey });
      if (sessionKeyRef.current !== deletingSessionKey) {
        return false;
      }
      await refetchModels();
      setForms((current) => ({
        ...current,
        [provider]: EMPTY_FORMS[provider],
      }));
      return true;
    } catch (error_) {
      if (!(error_ instanceof TypeError)) {
        throw error_;
      }
      setError(t('settings.credentialDeleteError'));
      return false;
    } finally {
      if (sessionKeyRef.current === deletingSessionKey) {
        setBusyProvider(null);
      }
    }
  };
  const hasPendingCredentials = providerList.some(
    ({ provider }) => forms[provider].apiKey.trim().length > 0,
  );

  return (
    <>
      <Dialog
        onOpenChange={onOpenChangeAction}
        open={open}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t('settings.credentialsTitle')}</DialogTitle>
            <DialogDescription>
              {t('settings.credentialsDescription')}
            </DialogDescription>
          </DialogHeader>
          <form
            className="grid gap-4"
            noValidate
            onSubmit={(event) => {
              void saveProviders(event);
            }}
          >
            <CredentialSettingsStatus
              loadError={credentialsLoadError}
              loading={loading}
              onRetryAction={() => {
                void refetch();
              }}
            />
            {!loading && !credentialsLoadError ? (
              <CredentialProviderList
                busyProvider={busyProvider}
                forms={forms}
                onDelete={setCredentialToDelete}
                onFieldChange={updateForm}
                saved={saved}
                saving={saving}
              />
            ) : null}
            {error === null ? null : (
              <p
                className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                role="alert"
              >
                {error}
              </p>
            )}
            <DialogFooter>
              <Button
                disabled={saving}
                onClick={() => {
                  onOpenChangeAction(false);
                }}
                type="button"
                variant="outline"
              >
                {t('common.cancel')}
              </Button>
              <Button
                aria-busy={saving || undefined}
                disabled={
                  saving || busyProvider !== null || !hasPendingCredentials
                }
                type="submit"
              >
                {saving ? <Spinner aria-hidden="true" /> : null}
                {saving ? t('composer.modelsLoading') : t('common.save')}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
      <CredentialDeleteDialog
        onConfirm={deleteProvider}
        onOpenChange={(isDeleteOpen) => {
          if (!isDeleteOpen) {
            setCredentialToDelete(null);
          }
        }}
        provider={credentialToDelete}
      />
    </>
  );
};

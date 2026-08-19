'use client';

import { useQueryClient } from '@tanstack/react-query';
import {
  type SyntheticEvent,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import { CredentialDeleteDialog } from '@/components/shell/credential-delete-dialog';
import {
  beginCredentialMutation,
  finishCredentialMutation,
  hasPendingCredentialMutation,
  subscribeCredentialMutations,
} from '@/components/shell/credential-mutation-coordinator';
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
  readonly onRestoreFocusAction?: () => boolean;
  readonly open: boolean;
};

type DialogOperation = {
  readonly dialogCycle: number;
  readonly sessionKey: string;
};

const providerList: readonly ProviderConfig[] = PROVIDERS;
const SHORT_DIALOG_MAX_HEIGHT_PX = 480;
const useIsomorphicLayoutEffect =
  typeof window === 'undefined' ? useEffect : useLayoutEffect;

type CredentialProviderListProps = {
  readonly busyProvider: ChatCredentialProvider | null;
  readonly failures: readonly CredentialSaveFailure[];
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

const saveFailureMessage = (kind: CredentialSaveFailure['kind']): string => {
  switch (kind) {
    case 'base-url':
      return t('settings.credentialBaseUrlError');
    case 'save':
      return t('settings.credentialSaveError');
    default:
      return kind satisfies never;
  }
};

const failureKinds = ['base-url', 'save'] as const satisfies ReadonlyArray<
  CredentialSaveFailure['kind']
>;

const saveFailureSummary = (
  failures: readonly CredentialSaveFailure[],
): string => {
  const providersByKind: Record<CredentialSaveFailure['kind'], string[]> = {
    'base-url': [],
    save: [],
  };
  for (const failure of failures) {
    const provider = providerList.find(
      (entry) => entry.provider === failure.provider,
    );
    const providerLabel =
      provider === undefined ? failure.provider : t(provider.labelKey);
    providersByKind[failure.kind].push(providerLabel);
  }
  return failureKinds
    .flatMap((kind) => {
      const providers = providersByKind[kind];
      return providers.length === 0
        ? []
        : [`${providers.join(', ')}: ${saveFailureMessage(kind)}`];
    })
    .join(' ');
};

const withoutProviderFailures = (
  failures: readonly CredentialSaveFailure[],
  provider: ChatCredentialProvider,
): readonly CredentialSaveFailure[] =>
  failures.filter((failure) => failure.provider !== provider);

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
  failures,
  forms,
  onDelete,
  onFieldChange,
  saved,
  saving,
}: CredentialProviderListProps) => (
  <div className="flex flex-col gap-3">
    {providerList.map(({ keyUrl, labelKey, provider }) => (
      <CredentialProviderForm
        busy={saving || busyProvider === provider}
        credential={saved[provider]}
        failure={failures.find((failure) => failure.provider === provider)}
        form={forms[provider]}
        key={provider}
        keyUrl={keyUrl}
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
  onRestoreFocusAction,
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
  const [dialogContent, setDialogContent] = useState<HTMLDivElement | null>(
    null,
  );
  const [error, setError] = useState<null | string>(null);
  const [shortDialogHeight, setShortDialogHeight] = useState(false);
  const [saveFailures, setSaveFailures] = useState<
    readonly CredentialSaveFailure[]
  >([]);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const providerScrollerRef = useRef<HTMLDivElement>(null);
  const revealSaveErrorRef = useRef(false);
  const dialogCycleRef = useRef(0);
  const lifecycleRef = useRef({ open, sessionKey });
  const sessionKeyRef = useRef(sessionKey);
  const saving = useSyncExternalStore(
    subscribeCredentialMutations,
    () => hasPendingCredentialMutation(sessionKey),
    () => false,
  );
  useIsomorphicLayoutEffect(() => {
    let observer: null | ResizeObserver = null;
    if (open && dialogContent !== null) {
      const update = () => {
        const availableHeight = Number.parseFloat(
          getComputedStyle(dialogContent).maxHeight,
        );
        setShortDialogHeight(
          Number.isFinite(availableHeight) &&
            availableHeight <= SHORT_DIALOG_MAX_HEIGHT_PX,
        );
      };
      update();
      if (typeof ResizeObserver === 'function') {
        observer = new ResizeObserver(update);
        observer.observe(dialogContent);
      }
    }
    return () => {
      observer?.disconnect();
    };
  }, [dialogContent, open]);
  useIsomorphicLayoutEffect(() => {
    const previous = lifecycleRef.current;
    sessionKeyRef.current = sessionKey;
    if (previous.open !== open || previous.sessionKey !== sessionKey) {
      dialogCycleRef.current += 1;
      lifecycleRef.current = { open, sessionKey };
    }
  }, [open, sessionKey]);
  const runForCurrentDialogOperation = (
    operation: DialogOperation,
    update: () => void,
  ) => {
    if (
      sessionKeyRef.current === operation.sessionKey &&
      dialogCycleRef.current === operation.dialogCycle
    ) {
      update();
    }
  };
  useEffect(() => {
    revealSaveErrorRef.current = false;
    setBusyProvider(null);
    setCredentialToDelete(null);
    setError(null);
    setSaveFailures([]);
    setForms(EMPTY_FORMS);
  }, [sessionKey]);
  useEffect(() => {
    if (!open) {
      revealSaveErrorRef.current = false;
      setCredentialToDelete(null);
      setError(null);
      setSaveFailures([]);
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
    setSaveFailures((current) =>
      current.filter(
        (failure) => failure.provider !== provider || failure.field !== field,
      ),
    );
    setForms((current) => ({
      ...current,
      [provider]: { ...current[provider], [field]: value },
    }));
  };
  const saveProviders = async (event: SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (sessionKey === null || !beginCredentialMutation(sessionKey)) {
      return;
    }
    const operation = {
      dialogCycle: dialogCycleRef.current,
      sessionKey,
    } satisfies DialogOperation;
    revealSaveErrorRef.current = false;
    setError(null);
    setSaveFailures([]);
    try {
      const {
        credentials: savedCredentials,
        failures,
        unexpectedError,
      } = await saveEnteredCredentials(forms);
      if (sessionKeyRef.current !== operation.sessionKey) {
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
        if (sessionKeyRef.current !== operation.sessionKey) {
          return;
        }
        await refetchModels();
        runForCurrentDialogOperation(operation, () => {
          setForms((current) =>
            formsWithSavedCredentials(current, savedCredentials),
          );
        });
      }
      runForCurrentDialogOperation(operation, () => {
        revealSaveErrorRef.current = failures.length > 0;
        setSaveFailures(failures);
      });
      if (unexpectedError !== null) {
        reportError(unexpectedError.reason);
      }
    } catch (error_) {
      if (error_ instanceof TypeError) {
        runForCurrentDialogOperation(operation, () => {
          revealSaveErrorRef.current = true;
          setError(t('settings.credentialSaveError'));
        });
      } else {
        throw error_;
      }
    } finally {
      finishCredentialMutation(operation.sessionKey);
    }
  };

  const deleteProvider = async (
    provider: ChatCredentialProvider,
  ): Promise<boolean> => {
    if (sessionKey === null || !beginCredentialMutation(sessionKey)) {
      return false;
    }
    const operation = {
      dialogCycle: dialogCycleRef.current,
      sessionKey,
    } satisfies DialogOperation;
    setBusyProvider(provider);
    revealSaveErrorRef.current = false;
    setError(null);
    try {
      const deleted = await deleteCredential(provider);
      if (!deleted) {
        runForCurrentDialogOperation(operation, () => {
          revealSaveErrorRef.current = true;
          setError(t('settings.credentialDeleteError'));
        });
        return false;
      }
      if (sessionKeyRef.current !== operation.sessionKey) {
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
      if (sessionKeyRef.current !== operation.sessionKey) {
        return false;
      }
      await refetchModels();
      if (sessionKeyRef.current !== operation.sessionKey) {
        return false;
      }
      runForCurrentDialogOperation(operation, () => {
        setForms((current) => ({
          ...current,
          [provider]: EMPTY_FORMS[provider],
        }));
        setSaveFailures((current) =>
          withoutProviderFailures(current, provider),
        );
      });
      return true;
    } catch (error_) {
      if (!(error_ instanceof TypeError)) {
        throw error_;
      }
      runForCurrentDialogOperation(operation, () => {
        revealSaveErrorRef.current = true;
        setError(t('settings.credentialDeleteError'));
      });
      return false;
    } finally {
      finishCredentialMutation(operation.sessionKey);
      runForCurrentDialogOperation(operation, () => {
        setBusyProvider(null);
      });
    }
  };
  const hasPendingCredentials = providerList.some(
    ({ provider }) => forms[provider].apiKey.trim().length > 0,
  );
  const saveError =
    error ??
    (saveFailures.length === 0 ? null : saveFailureSummary(saveFailures));
  useIsomorphicLayoutEffect(() => {
    if (saveError === null) {
      revealSaveErrorRef.current = false;
      return;
    }
    if (
      revealSaveErrorRef.current &&
      shortDialogHeight &&
      providerScrollerRef.current !== null
    ) {
      providerScrollerRef.current.scrollTop = 0;
      revealSaveErrorRef.current = false;
    }
  }, [saveError, shortDialogHeight]);

  return (
    <>
      <Dialog
        onOpenChange={onOpenChangeAction}
        open={open}
      >
        <DialogContent
          className="grid grid-rows-[auto_minmax(0,1fr)] gap-0 overflow-clip p-0 sm:max-w-2xl"
          onCloseAutoFocus={(event) => {
            if (onRestoreFocusAction?.() === true) {
              event.preventDefault();
            }
          }}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            titleRef.current?.focus();
          }}
          ref={setDialogContent}
        >
          <DialogHeader className="border-b border-border px-6 py-5 pr-14">
            <DialogTitle
              className="outline-none"
              onKeyDown={(event) => {
                if (event.key !== 'Tab' || !event.shiftKey) {
                  return;
                }
                const closeButton = event.currentTarget
                  .closest('[role="dialog"]')
                  ?.querySelector<HTMLElement>('[data-slot="dialog-close"]');
                if (closeButton === null || closeButton === undefined) {
                  return;
                }
                event.preventDefault();
                closeButton.focus();
              }}
              ref={titleRef}
              tabIndex={-1}
            >
              {t('settings.credentialsTitle')}
            </DialogTitle>
            <DialogDescription>
              {t('settings.credentialsDescription')}
            </DialogDescription>
          </DialogHeader>
          <form
            className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto]"
            noValidate
            onSubmit={(event) => {
              void saveProviders(event);
            }}
          >
            <div
              className="min-h-0 overflow-y-auto p-4 sm:p-6"
              ref={providerScrollerRef}
            >
              <div className="grid gap-4">
                <CredentialSettingsStatus
                  loadError={credentialsLoadError}
                  loading={loading}
                  onRetryAction={() => {
                    void refetch();
                  }}
                />
                {saveError === null || !shortDialogHeight ? null : (
                  <p
                    className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-pretty text-sm text-destructive"
                    role="alert"
                  >
                    {saveError}
                  </p>
                )}
                {!loading && !credentialsLoadError ? (
                  <CredentialProviderList
                    busyProvider={busyProvider}
                    failures={saveFailures}
                    forms={forms}
                    onDelete={setCredentialToDelete}
                    onFieldChange={updateForm}
                    saved={saved}
                    saving={saving}
                  />
                ) : null}
              </div>
            </div>
            <div className="border-t border-border bg-background">
              {saveError === null || shortDialogHeight ? null : (
                <p
                  className="mx-4 mt-4 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-pretty text-sm text-destructive sm:mx-6"
                  role="alert"
                >
                  {saveError}
                </p>
              )}
              <DialogFooter
                className={
                  shortDialogHeight
                    ? 'flex-row p-4 sm:px-6'
                    : 'flex-col p-4 sm:px-6'
                }
              >
                <Button
                  className={
                    shortDialogHeight ? 'w-auto flex-1' : 'w-full sm:w-auto'
                  }
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
                  className={
                    shortDialogHeight
                      ? 'w-auto flex-1 disabled:opacity-70'
                      : 'w-full disabled:opacity-70 sm:w-auto'
                  }
                  disabled={
                    saving || busyProvider !== null || !hasPendingCredentials
                  }
                  type="submit"
                >
                  {saving ? <Spinner aria-hidden="true" /> : null}
                  {saving
                    ? t('settings.savingCredentials')
                    : t('settings.saveCredentials')}
                </Button>
              </DialogFooter>
            </div>
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

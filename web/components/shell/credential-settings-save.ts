import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import {
  CredentialBaseUrlRejectedError,
  saveCredential,
} from '@/components/shell/credential-settings-client';
import {
  type ProviderForm,
  PROVIDERS,
} from '@/components/shell/credential-settings-data';

export type CredentialSaveFailure = 'base-url' | 'save';

type CredentialForms = Readonly<Record<ChatCredentialProvider, ProviderForm>>;

type CredentialSaveBatch = {
  readonly credentials: readonly ChatCredentialPublic[];
  readonly failure: CredentialSaveFailure | null;
  readonly unexpectedError: null | { readonly reason: unknown };
};

export const saveEnteredCredentials = async (
  forms: CredentialForms,
): Promise<CredentialSaveBatch> => {
  const pendingCredentials = PROVIDERS.flatMap(({ provider }) => {
    const form = forms[provider];
    const apiKey = form.apiKey.trim();
    return apiKey.length === 0
      ? []
      : [{ apiKey, baseUrl: form.baseUrl.trim(), provider }];
  });
  const results = await Promise.allSettled(
    pendingCredentials.map((credential) => saveCredential(credential)),
  );
  const credentials: ChatCredentialPublic[] = [];
  let failure: CredentialSaveFailure | null = null;
  let unexpectedError: CredentialSaveBatch['unexpectedError'] = null;
  for (const result of results) {
    if (result.status === 'fulfilled') {
      if (result.value === null) {
        failure ??= 'save';
      } else {
        credentials.push(result.value);
      }
      continue;
    }
    const reason: unknown = result.reason;
    if (reason instanceof CredentialBaseUrlRejectedError) {
      failure = 'base-url';
    } else if (reason instanceof TypeError) {
      failure ??= 'save';
    } else if (unexpectedError === null) {
      failure ??= 'save';
      unexpectedError = { reason };
    }
  }
  return { credentials, failure, unexpectedError };
};

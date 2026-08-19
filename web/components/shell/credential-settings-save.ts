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

export type CredentialSaveFailure =
  | {
      readonly field: 'apiKey';
      readonly kind: 'save';
      readonly provider: ChatCredentialProvider;
    }
  | {
      readonly field: 'baseUrl';
      readonly kind: 'base-url';
      readonly provider: ChatCredentialProvider;
    };

type CredentialForms = Readonly<Record<ChatCredentialProvider, ProviderForm>>;

type CredentialSaveBatch = {
  readonly credentials: readonly ChatCredentialPublic[];
  readonly failures: readonly CredentialSaveFailure[];
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
  const failures: CredentialSaveFailure[] = [];
  let unexpectedError: CredentialSaveBatch['unexpectedError'] = null;
  for (const [index, result] of results.entries()) {
    const pendingCredential = pendingCredentials.at(index);
    if (pendingCredential === undefined) {
      continue;
    }
    if (result.status === 'fulfilled') {
      if (result.value === null) {
        failures.push({
          field: 'apiKey',
          kind: 'save',
          provider: pendingCredential.provider,
        });
      } else {
        credentials.push(result.value);
      }
      continue;
    }
    const reason: unknown = result.reason;
    const failure: CredentialSaveFailure =
      reason instanceof CredentialBaseUrlRejectedError
        ? {
            field: 'baseUrl',
            kind: 'base-url',
            provider: pendingCredential.provider,
          }
        : {
            field: 'apiKey',
            kind: 'save',
            provider: pendingCredential.provider,
          };
    failures.push(failure);
    if (
      !(reason instanceof CredentialBaseUrlRejectedError) &&
      !(reason instanceof TypeError) &&
      unexpectedError === null
    ) {
      unexpectedError = { reason };
    }
  }
  return { credentials, failures, unexpectedError };
};

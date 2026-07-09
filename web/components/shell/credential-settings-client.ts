import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import {
  isCredential,
  parseCredentials,
} from '@/components/shell/credential-settings-data';

type SaveCredentialParams = {
  readonly apiKey: string;
  readonly baseUrl: string;
  readonly provider: ChatCredentialProvider;
};

const API_KEY_FIELD = 'api_key';
const BASE_URL_FIELD = 'base_url';

export const loadCredentials = async (
  signal: AbortSignal,
): Promise<null | readonly ChatCredentialPublic[]> => {
  const response = await fetch('/api/chat/credentials', { signal });
  if (!response.ok) {
    return null;
  }
  const body: unknown = await response.json();
  return parseCredentials(body);
};

export const saveCredential = async ({
  apiKey,
  baseUrl,
  provider,
}: SaveCredentialParams): Promise<ChatCredentialPublic | null> => {
  const response = await fetch('/api/chat/credentials', {
    body: JSON.stringify({
      [API_KEY_FIELD]: apiKey,
      [BASE_URL_FIELD]: baseUrl || null,
      provider,
    }),
    headers: { 'content-type': 'application/json' },
    method: 'PUT',
  });
  if (!response.ok) {
    return null;
  }
  const body: unknown = await response.json();
  return isCredential(body) ? body : null;
};

export const deleteCredential = async (
  provider: ChatCredentialProvider,
): Promise<boolean> => {
  const response = await fetch(`/api/chat/credentials/${provider}`, {
    method: 'DELETE',
  });
  return response.ok;
};

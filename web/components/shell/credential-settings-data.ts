import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';
import type { TKey } from '@/lib/i18n';

export type ProviderConfig = {
  readonly labelKey: TKey;
  readonly provider: ChatCredentialProvider;
};

export type ProviderForm = {
  readonly apiKey: string;
  readonly baseUrl: string;
};

export const PROVIDERS = [
  { labelKey: 'settings.provider.openai', provider: 'openai' },
  { labelKey: 'settings.provider.google', provider: 'google' },
  { labelKey: 'settings.provider.anthropic', provider: 'anthropic' },
] as const satisfies readonly ProviderConfig[];

export const EMPTY_FORMS = {
  anthropic: { apiKey: '', baseUrl: '' },
  google: { apiKey: '', baseUrl: '' },
  openai: { apiKey: '', baseUrl: '' },
} satisfies Record<ChatCredentialProvider, ProviderForm>;

const PROVIDER_SET: ReadonlySet<unknown> = new Set(
  PROVIDERS.map(({ provider }) => provider),
);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const isProvider = (value: unknown): value is ChatCredentialProvider =>
  PROVIDER_SET.has(value);

export const isCredential = (value: unknown): value is ChatCredentialPublic => {
  if (!isRecord(value) || !isProvider(value['provider'])) {
    return false;
  }
  const baseUrl = value['base_url'];
  return baseUrl === null || typeof baseUrl === 'string';
};

export const credentialsByProvider = (
  credentials: readonly ChatCredentialPublic[],
): Partial<Record<ChatCredentialProvider, ChatCredentialPublic>> => {
  const byProvider: Partial<
    Record<ChatCredentialProvider, ChatCredentialPublic>
  > = {};
  for (const credential of credentials) {
    byProvider[credential.provider] = credential;
  }
  return byProvider;
};

export const parseCredentials = (
  value: unknown,
): readonly ChatCredentialPublic[] =>
  Array.isArray(value) ? value.filter(isCredential) : [];

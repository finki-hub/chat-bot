import type {
  CatalogProvider,
  CatalogSource,
  ModelAvailability,
  ModelCatalog,
  ModelDescriptor,
  SponsoredQuota,
} from '@/lib/api-types';

import { t, type TKey } from '@/lib/i18n';

/* eslint-disable camelcase -- catalog fields mirror the Python wire contract. */

const PROVIDER_LABEL_KEYS: Record<CatalogProvider, TKey> = {
  anthropic: 'settings.provider.anthropic',
  google: 'settings.provider.google',
  ollama: 'settings.provider.ollama',
  openai: 'settings.provider.openai',
};

const CATALOG_PROVIDERS = ['anthropic', 'google', 'ollama', 'openai'] as const;
const API_CATALOG_SOURCES = ['live', 'snapshot', 'stale'] as const;
const MODEL_AVAILABILITIES = [
  'both',
  'byok',
  'sponsored',
  'unavailable',
] as const;

const CURATED_MODEL_DATA = [
  ['gpt-5.6-sol', 'GPT-5.6 Sol', 'openai'],
  ['gpt-5.6-terra', 'GPT-5.6 Terra', 'openai'],
  ['gpt-5.6-luna', 'GPT-5.6 Luna', 'openai'],
  ['gpt-5.5', 'GPT-5.5', 'openai'],
  ['gpt-5.4', 'GPT-5.4', 'openai'],
  ['gpt-5.4-mini', 'GPT-5.4 Mini', 'openai'],
  ['gpt-5.4-nano', 'GPT-5.4 Nano', 'openai'],
  ['gemini-3.1-pro-preview', 'Gemini 3.1 Pro Preview', 'google'],
  ['gemini-3.5-flash', 'Gemini 3.5 Flash', 'google'],
  ['gemini-3.1-flash-lite', 'Gemini 3.1 Flash Lite', 'google'],
  ['claude-opus-4-8', 'Claude Opus 4.8', 'anthropic'],
  ['claude-sonnet-5', 'Claude Sonnet 5', 'anthropic'],
  ['claude-haiku-4-5', 'Claude Haiku 4.5', 'anthropic'],
  ['qwen3:30b-a3b-thinking-2507-q4_K_M', 'Qwen3 30B Thinking', 'ollama'],
  ['qwen3:30b-a3b-instruct-2507-q4_K_M', 'Qwen3 30B Instruct', 'ollama'],
  ['qwen3:14b-q4_K_M', 'Qwen3 14B', 'ollama'],
] as const satisfies ReadonlyArray<readonly [string, string, CatalogProvider]>;

export const CURATED_MODEL_DESCRIPTORS: Readonly<
  Record<string, ModelDescriptor>
> = Object.fromEntries(
  CURATED_MODEL_DATA.map(([id, name, provider]) => [
    id,
    { id, name, provider },
  ]),
);

const PROVIDER_PREFIXES: ReadonlyArray<readonly [string, CatalogProvider]> = [
  ['gpt', 'openai'],
  ['text-embedding', 'openai'],
  ['gemini', 'google'],
  ['claude', 'anthropic'],
  ['llama', 'ollama'],
  ['deepseek', 'ollama'],
  ['mistral', 'ollama'],
  ['qwen', 'ollama'],
  ['hf.co/', 'ollama'],
];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isCatalogProvider = (value: unknown): value is CatalogProvider =>
  CATALOG_PROVIDERS.includes(value as CatalogProvider);

const isCatalogSource = (value: unknown): value is CatalogSource =>
  API_CATALOG_SOURCES.includes(value as (typeof API_CATALOG_SOURCES)[number]);

const isModelAvailability = (value: unknown): value is ModelAvailability =>
  MODEL_AVAILABILITIES.includes(value as ModelAvailability);

const inferProvider = (id: string): string => {
  const lower = id.toLowerCase();
  for (const [prefix, provider] of PROVIDER_PREFIXES) {
    if (lower.startsWith(prefix)) {
      return provider;
    }
  }
  const slash = id.indexOf('/');
  if (slash > 0) {
    return id.slice(0, slash);
  }
  const dash = id.indexOf('-');
  if (dash > 0) {
    return id.slice(0, dash);
  }
  return id.length > 0 ? id : 'other';
};

const isNonNegativeInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

const normalizeSponsoredQuota = (value: unknown): null | SponsoredQuota => {
  if (!isRecord(value)) {
    return null;
  }

  const { limit, remaining, resets_at: resetsAt } = value;
  if (
    !isNonNegativeInteger(limit) ||
    !isNonNegativeInteger(remaining) ||
    remaining > limit ||
    typeof resetsAt !== 'string' ||
    resetsAt.length === 0
  ) {
    return null;
  }

  return { limit, remaining, resets_at: resetsAt };
};

const normalizeDescriptor = (
  value: unknown,
  typedEnvelope = false,
): ModelDescriptor | null => {
  if (!isRecord(value)) {
    return null;
  }
  const {
    availability,
    description,
    id,
    loaded,
    name,
    provider,
    sponsored_quota: sponsoredQuota,
  } = value;
  if (typeof id !== 'string' || id.length === 0) {
    return null;
  }
  if (
    typedEnvelope &&
    (typeof name !== 'string' ||
      name.length === 0 ||
      !isCatalogProvider(provider) ||
      (loaded !== undefined &&
        loaded !== null &&
        typeof loaded !== 'boolean') ||
      (description !== undefined &&
        description !== null &&
        typeof description !== 'string'))
  ) {
    return null;
  }
  const base = {
    id,
    ...((loaded === null || typeof loaded === 'boolean') && { loaded }),
    name: typeof name === 'string' && name.length > 0 ? name : id,
    provider: isCatalogProvider(provider) ? provider : inferProvider(id),
  } satisfies ModelDescriptor;
  const normalizedQuota = normalizeSponsoredQuota(sponsoredQuota);
  return {
    ...base,
    ...(typeof availability === 'string' &&
      isModelAvailability(availability) && { availability }),
    ...(typeof description === 'string' &&
      description.length > 0 && { description }),
    ...(normalizedQuota !== null && { sponsored_quota: normalizedQuota }),
  };
};

export type ModelGroup = {
  readonly models: readonly ModelDescriptor[];
  readonly provider: string;
};

export const parseModelCatalog = (value: unknown): ModelCatalog => {
  if (isRecord(value)) {
    const { models: rawModels, source: rawSource, version } = value;
    if (
      version === 1 &&
      isCatalogSource(rawSource) &&
      Array.isArray(rawModels) &&
      rawModels.length > 0
    ) {
      const models = rawModels.map((entry) => normalizeDescriptor(entry, true));
      if (models.every((model): model is ModelDescriptor => model !== null)) {
        return { models, source: rawSource, version: 1 };
      }
    }
    return { models: [], source: 'error', version: 1 };
  }

  if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    const models = value
      .map((id) => CURATED_MODEL_DESCRIPTORS[id] ?? normalizeDescriptor({ id }))
      .filter((model): model is ModelDescriptor => model !== null);
    return { models, source: 'live', version: 1 };
  }

  return { models: [], source: 'error', version: 1 };
};

export const groupModelsByProvider = (
  models: readonly ModelDescriptor[],
): ModelGroup[] => {
  const buckets = new Map<string, ModelDescriptor[]>();
  for (const model of models) {
    const providerBucket = buckets.get(model.provider);
    if (providerBucket) {
      providerBucket.push(model);
    } else {
      buckets.set(model.provider, [model]);
    }
  }
  return Array.from(buckets, ([provider, providerModels]) => ({
    models: providerModels,
    provider,
  }));
};

export const isSponsoredModel = (
  model: Pick<ModelDescriptor, 'availability'>,
): boolean =>
  model.availability === 'both' || model.availability === 'sponsored';

export const isModelAvailable = (
  model: Pick<ModelDescriptor, 'availability' | 'provider'>,
  availableProviders: ReadonlySet<string>,
): boolean =>
  isSponsoredModel(model) ||
  (model.availability !== 'unavailable' &&
    availableProviders.has(model.provider));

// Repair a persisted selection when the catalog no longer contains it: prefer the current
// model, then the default, then the first available model, and only keep an unknown id when
// the catalog is empty (so a transient miss never wipes it).
export const recoverSelectedModel = (
  models: readonly ModelDescriptor[],
  current: string,
  fallback: string,
): string => {
  if (models.some((model) => model.id === current)) {
    return current;
  }
  if (models.some((model) => model.id === fallback)) {
    return fallback;
  }
  return models[0]?.id ?? current;
};

export const providerLabel = (provider: string): string =>
  isCatalogProvider(provider) ? t(PROVIDER_LABEL_KEYS[provider]) : provider;

/* eslint-enable camelcase -- end wire-contract fields. */

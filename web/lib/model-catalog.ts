import type {
  CatalogProvider,
  CatalogSource,
  CatalogTier,
  ModelCatalog,
  ModelDescriptor,
} from '@/lib/api-types';

import { t, type TKey } from '@/lib/i18n';

const PROVIDER_LABEL_KEYS: Record<CatalogProvider, TKey> = {
  anthropic: 'settings.provider.anthropic',
  google: 'settings.provider.google',
  ollama: 'settings.provider.ollama',
  openai: 'settings.provider.openai',
};

const TIER_LABEL_KEYS: Record<CatalogTier, TKey> = {
  cheap: 'composer.tier.cheap',
  default: 'composer.tier.default',
  premium: 'composer.tier.premium',
};

const CATALOG_PROVIDERS = ['anthropic', 'google', 'ollama', 'openai'] as const;
const CATALOG_TIERS = ['cheap', 'default', 'premium'] as const;
const TIER_ORDER = ['premium', 'default', 'cheap'] as const;
const API_CATALOG_SOURCES = ['live', 'snapshot', 'stale'] as const;

const CURATED_MODEL_DATA = [
  ['gpt-5.4', 'GPT-5.4', 'openai', 'premium'],
  ['gpt-5.4-mini', 'GPT-5.4 Mini', 'openai', 'default'],
  ['gpt-5.4-nano', 'GPT-5.4 Nano', 'openai', 'cheap'],
  ['gemini-2.5-pro', 'Gemini 2.5 Pro', 'google', 'premium'],
  ['gemini-2.5-flash', 'Gemini 2.5 Flash', 'google', 'default'],
  ['claude-opus-4-8', 'Claude Opus 4.8', 'anthropic', 'premium'],
  ['claude-sonnet-5', 'Claude Sonnet 5', 'anthropic', 'default'],
  ['claude-haiku-4-5', 'Claude Haiku 4.5', 'anthropic', 'cheap'],
  ['llama3.3:70b', 'Llama 3.3 70B', 'ollama', 'default'],
  ['deepseek-r1:70b', 'DeepSeek R1 70B', 'ollama', 'default'],
  [
    'hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0',
    'Domestic Yak 8B Instruct',
    'ollama',
    'cheap',
  ],
  ['hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0', 'VezilkaLLM', 'ollama', 'cheap'],
] as const satisfies ReadonlyArray<
  readonly [string, string, CatalogProvider, CatalogTier]
>;

export const CURATED_MODEL_DESCRIPTORS: Readonly<
  Record<string, ModelDescriptor>
> = Object.fromEntries(
  CURATED_MODEL_DATA.map(([id, name, provider, tier]) => [
    id,
    { id, name, provider, tier },
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

const isCatalogTier = (value: unknown): value is CatalogTier =>
  CATALOG_TIERS.includes(value as CatalogTier);

const isCatalogSource = (value: unknown): value is CatalogSource =>
  API_CATALOG_SOURCES.includes(value as (typeof API_CATALOG_SOURCES)[number]);

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

const normalizeDescriptor = (
  value: unknown,
  typedEnvelope = false,
): ModelDescriptor | null => {
  if (!isRecord(value)) {
    return null;
  }
  const { description, id, name, provider, tier } = value;
  if (typeof id !== 'string' || id.length === 0) {
    return null;
  }
  if (
    typedEnvelope &&
    (typeof name !== 'string' ||
      name.length === 0 ||
      !isCatalogProvider(provider) ||
      !isCatalogTier(tier) ||
      (description !== undefined &&
        description !== null &&
        typeof description !== 'string'))
  ) {
    return null;
  }
  const base = {
    id,
    name: typeof name === 'string' && name.length > 0 ? name : id,
    provider: isCatalogProvider(provider) ? provider : inferProvider(id),
    tier: isCatalogTier(tier) ? tier : 'default',
  } satisfies ModelDescriptor;
  return typeof description === 'string' && description.length > 0
    ? { ...base, description }
    : base;
};

export type ModelGroup = {
  readonly providers: readonly ModelProviderGroup[];
  readonly tier: CatalogTier;
};

export type ModelProviderGroup = {
  readonly models: readonly ModelDescriptor[];
  readonly provider: string;
};

export const parseModelCatalog = (value: unknown): ModelCatalog => {
  if (isRecord(value)) {
    const { models: rawModels, source: rawSource, version } = value;
    if (
      version === 1 &&
      isCatalogSource(rawSource) &&
      Array.isArray(rawModels)
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

export const groupModelsByProviderTier = (
  models: readonly ModelDescriptor[],
): ModelGroup[] => {
  const buckets = new Map<CatalogTier, Map<string, ModelDescriptor[]>>();
  for (const model of models) {
    const tierBucket = buckets.get(model.tier);
    if (tierBucket) {
      const providerBucket = tierBucket.get(model.provider);
      if (providerBucket) {
        providerBucket.push(model);
      } else {
        tierBucket.set(model.provider, [model]);
      }
    } else {
      buckets.set(model.tier, new Map([[model.provider, [model]]]));
    }
  }
  return TIER_ORDER.flatMap((tier) => {
    const tierBucket = buckets.get(tier);
    return tierBucket
      ? [
          {
            providers: Array.from(tierBucket, ([provider, providerModels]) => ({
              models: providerModels,
              provider,
            })),
            tier,
          },
        ]
      : [];
  });
};

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

export const tierLabel = (tier: CatalogTier): string =>
  t(TIER_LABEL_KEYS[tier]);

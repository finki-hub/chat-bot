import { describe, expect, it } from 'vitest';

import type { CatalogTier, ModelDescriptor } from '@/lib/api-types';

import {
  groupModelsByProviderTier,
  parseModelCatalog,
  providerLabel,
  recoverSelectedModel,
} from '@/lib/model-catalog';

const OPENAI = 'openai';
const GOOGLE = 'google';
const ANTHROPIC = 'anthropic';
const OLLAMA = 'ollama';
const PREMIUM = 'premium';
const STANDARD = 'default';
const CHEAP = 'cheap';
const LIVE = 'live';

const GPT = 'gpt-5.4';
const GPT_MINI = 'gpt-5.4-mini';
const GPT_MINI_NAME = 'GPT-5.4 Mini';
const GPT_NANO = 'gpt-5.4-nano';
const GEMINI_PRO = 'gemini-2.5-pro';
const GEMINI_FLASH = 'gemini-2.5-flash';
const CLAUDE_OPUS = 'claude-opus-4-8';
const CLAUDE_5 = 'claude-sonnet-5';
const CLAUDE_LEGACY = 'claude-sonnet-4-6';
const LLAMA = 'llama3.3:70b';
const DEEPSEEK = 'deepseek-r1:70b';
const DOMESTIC_YAK = 'hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0';
const VEZILKA = 'hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0';

const descriptor = (
  id: string,
  provider: string,
  tier: CatalogTier,
): ModelDescriptor => ({ id, name: id, provider, tier });

const typedModels = [
  {
    capabilities: { reasoning: true },
    description: 'OpenAI flagship reasoning model.',
    id: GPT,
    name: 'GPT-5.4',
    pricing: { input: 2.5, output: 15 },
    provider: OPENAI,
    tier: PREMIUM,
  },
  { id: GPT_NANO, name: 'GPT-5.4 Nano', provider: OPENAI, tier: CHEAP },
  { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI, tier: STANDARD },
  { id: GEMINI_PRO, name: 'Gemini 2.5 Pro', provider: GOOGLE, tier: PREMIUM },
  {
    id: CLAUDE_OPUS,
    name: 'Claude Opus 4.8',
    provider: ANTHROPIC,
    tier: PREMIUM,
  },
  { id: LLAMA, name: 'Llama 3.3 70B', provider: OLLAMA, tier: STANDARD },
];

const typedCatalog = (source: string = LIVE) => ({
  models: typedModels,
  source,
  version: 1,
});

describe('parseModelCatalog', () => {
  it('parses a typed catalog and keeps only the web-relevant fields', () => {
    const catalog = parseModelCatalog(typedCatalog());

    expect(catalog.source).toBe(LIVE);
    expect(catalog.version).toBe(1);
    expect(catalog.models).toHaveLength(6);
    expect(catalog.models[0]).toStrictEqual({
      description: 'OpenAI flagship reasoning model.',
      id: GPT,
      name: 'GPT-5.4',
      provider: OPENAI,
      tier: PREMIUM,
    });
  });

  it('normalizes every curated legacy id with its immutable fallback descriptor', () => {
    const legacyIds = [
      GPT,
      GPT_MINI,
      GPT_NANO,
      GEMINI_PRO,
      GEMINI_FLASH,
      CLAUDE_OPUS,
      CLAUDE_5,
      'claude-haiku-4-5',
      LLAMA,
      DEEPSEEK,
      DOMESTIC_YAK,
      VEZILKA,
    ];
    const catalog = parseModelCatalog(legacyIds);

    expect(catalog.source).toBe(LIVE);
    expect(catalog.models).toStrictEqual([
      { id: GPT, name: 'GPT-5.4', provider: OPENAI, tier: PREMIUM },
      { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI, tier: STANDARD },
      { id: GPT_NANO, name: 'GPT-5.4 Nano', provider: OPENAI, tier: CHEAP },
      {
        id: GEMINI_PRO,
        name: 'Gemini 2.5 Pro',
        provider: GOOGLE,
        tier: PREMIUM,
      },
      {
        id: GEMINI_FLASH,
        name: 'Gemini 2.5 Flash',
        provider: GOOGLE,
        tier: STANDARD,
      },
      {
        id: CLAUDE_OPUS,
        name: 'Claude Opus 4.8',
        provider: ANTHROPIC,
        tier: PREMIUM,
      },
      {
        id: CLAUDE_5,
        name: 'Claude Sonnet 5',
        provider: ANTHROPIC,
        tier: STANDARD,
      },
      {
        id: 'claude-haiku-4-5',
        name: 'Claude Haiku 4.5',
        provider: ANTHROPIC,
        tier: CHEAP,
      },
      { id: LLAMA, name: 'Llama 3.3 70B', provider: OLLAMA, tier: STANDARD },
      {
        id: DEEPSEEK,
        name: 'DeepSeek R1 70B',
        provider: OLLAMA,
        tier: STANDARD,
      },
      {
        id: DOMESTIC_YAK,
        name: 'Domestic Yak 8B Instruct',
        provider: OLLAMA,
        tier: CHEAP,
      },
      { id: VEZILKA, name: 'VezilkaLLM', provider: OLLAMA, tier: CHEAP },
    ]);
  });

  it('uses inferred provider, raw id name, and default tier for an unknown legacy id', () => {
    const unknown = 'acme/new-model';

    expect(parseModelCatalog([unknown]).models).toStrictEqual([
      descriptor(unknown, 'acme', STANDARD),
    ]);
  });

  it.each([
    { models: typedModels, source: LIVE, version: 2 },
    { models: typedModels, source: 'weird', version: 1 },
    {
      models: [{ id: GPT_MINI, name: GPT_MINI_NAME, tier: STANDARD }],
      source: LIVE,
      version: 1,
    },
    {
      models: [
        {
          id: GPT_MINI,
          name: GPT_MINI_NAME,
          provider: OPENAI,
          tier: 'bogus',
        },
      ],
      source: LIVE,
      version: 1,
    },
  ])('rejects a malformed typed catalog envelope', (payload) => {
    expect(parseModelCatalog(payload)).toStrictEqual({
      models: [],
      source: 'error',
      version: 1,
    });
  });

  it('rejects a typed entry with no usable name', () => {
    const catalog = parseModelCatalog({
      models: [{ id: 'mystery-model', provider: OPENAI, tier: PREMIUM }],
      source: 'snapshot',
      version: 1,
    });

    expect(catalog).toStrictEqual({ models: [], source: 'error', version: 1 });
  });

  it('rejects the envelope when an entry has no string id', () => {
    const catalog = parseModelCatalog({
      models: [
        { name: 'No id', provider: OPENAI, tier: PREMIUM },
        { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI, tier: STANDARD },
      ],
      source: LIVE,
      version: 1,
    });

    expect(catalog).toStrictEqual({ models: [], source: 'error', version: 1 });
  });

  it('keeps a stale source reported by the BFF', () => {
    const catalog = parseModelCatalog(typedCatalog('stale'));

    expect(catalog.source).toBe('stale');
  });

  it('returns an empty error catalog for non-catalog junk', () => {
    const junkValues: unknown[] = [
      { models: 'oops' },
      null,
      42,
      'nope',
      undefined,
    ];

    for (const junk of junkValues) {
      const catalog = parseModelCatalog(junk);

      expect(catalog).toStrictEqual({
        models: [],
        source: 'error',
        version: 1,
      });
    }
  });

  it('treats an empty legacy list as a valid empty catalog', () => {
    expect(parseModelCatalog([])).toStrictEqual({
      models: [],
      source: LIVE,
      version: 1,
    });
  });
});

describe('groupModelsByProviderTier', () => {
  it('groups tier-first, then provider, while preserving API order within providers', () => {
    const { models } = parseModelCatalog(typedCatalog());

    const groups = groupModelsByProviderTier(models);

    expect(groups.map(({ tier }) => tier)).toStrictEqual([
      PREMIUM,
      STANDARD,
      CHEAP,
    ]);
    expect(
      groups.map(({ providers }) => providers.map(({ provider }) => provider)),
    ).toStrictEqual([[OPENAI, GOOGLE, ANTHROPIC], [OPENAI, OLLAMA], [OPENAI]]);
    expect(
      groups[0]?.providers[0]?.models.map((model) => model.id),
    ).toStrictEqual([GPT]);
    expect(
      groups[1]?.providers[0]?.models.map((model) => model.id),
    ).toStrictEqual([GPT_MINI]);
    expect(
      groups[2]?.providers[0]?.models.map((model) => model.id),
    ).toStrictEqual([GPT_NANO]);
  });

  it('is stable for models that share a tier', () => {
    const { models } = parseModelCatalog([LLAMA, DEEPSEEK]);

    const groups = groupModelsByProviderTier(models);

    expect(groups).toHaveLength(1);
    expect(groups[0]?.providers).toHaveLength(1);
    expect(
      groups[0]?.providers[0]?.models.map((model) => model.id),
    ).toStrictEqual([LLAMA, DEEPSEEK]);
  });
});

describe('recoverSelectedModel', () => {
  const { models } = parseModelCatalog(typedCatalog());

  it('keeps the current model when it is still in the catalog', () => {
    expect(recoverSelectedModel(models, GEMINI_PRO, GPT)).toBe(GEMINI_PRO);
  });

  it('falls back to the default when the current model is gone', () => {
    expect(recoverSelectedModel(models, CLAUDE_LEGACY, GPT_MINI)).toBe(
      GPT_MINI,
    );
  });

  it('falls back to the first model when neither current nor default exist', () => {
    expect(recoverSelectedModel(models, 'ghost', 'also-ghost')).toBe(GPT);
  });

  it('keeps the current model when the catalog is empty', () => {
    expect(recoverSelectedModel([], CLAUDE_LEGACY, GPT)).toBe(CLAUDE_LEGACY);
  });
});

describe('providerLabel', () => {
  it('maps known providers to their display label', () => {
    expect(providerLabel(OPENAI)).toBe('OpenAI');
    expect(providerLabel(ANTHROPIC)).toBe('Anthropic');
  });

  it('returns the raw provider for unknown buckets', () => {
    expect(providerLabel('BAAI')).toBe('BAAI');
  });
});

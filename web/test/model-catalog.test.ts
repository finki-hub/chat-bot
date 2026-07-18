import { describe, expect, it } from 'vitest';

/* eslint-disable camelcase -- fixtures mirror the Python catalog wire contract. */
import type { ModelDescriptor } from '@/lib/api-types';

import {
  CURATED_MODEL_DESCRIPTORS,
  groupModelsByProvider,
  isModelAvailable,
  isSponsoredModel,
  parseModelCatalog,
  providerLabel,
  recoverSelectedModel,
} from '@/lib/model-catalog';

const OPENAI = 'openai';
const GOOGLE = 'google';
const ANTHROPIC = 'anthropic';
const OLLAMA = 'ollama';
const LIVE = 'live';
const GPT = 'gpt-5.4';
const LUNA = 'gpt-5.6-luna';

const EXPECTED_CURATED_IDS = [
  'gpt-5.6-sol',
  'gpt-5.6-terra',
  LUNA,
  'gpt-5.5',
  'gpt-5.4',
  'gpt-5.4-mini',
  'gpt-5.4-nano',
  'gemini-3.1-pro-preview',
  'gemini-3.5-flash',
  'gemini-3.1-flash-lite',
  'claude-opus-4-8',
  'claude-sonnet-5',
  'claude-haiku-4-5',
  'qwen3:30b-a3b-thinking-2507-q4_K_M',
  'qwen3:30b-a3b-instruct-2507-q4_K_M',
  'qwen3:14b-q4_K_M',
];

const GPT_TERRA = 'gpt-5.6-terra';
const GPT_MINI = 'gpt-5.4-mini';
const GPT_MINI_NAME = 'GPT-5.4 Mini';
const GPT_NANO = 'gpt-5.4-nano';
const GEMINI_PRO = 'gemini-3.1-pro-preview';
const CLAUDE_OPUS = 'claude-opus-4-8';
const CLAUDE_LEGACY = 'claude-sonnet-4-6';
const QWEN_THINKING = 'qwen3:30b-a3b-thinking-2507-q4_K_M';

const descriptor = (id: string, provider: string): ModelDescriptor => ({
  id,
  name: id,
  provider,
});

const typedModels = [
  {
    capabilities: { reasoning: true },
    description: 'OpenAI flagship reasoning model.',
    id: GPT,
    name: 'GPT-5.4',
    pricing: { input: 2.5, output: 15 },
    provider: OPENAI,
  },
  { id: GPT_NANO, name: 'GPT-5.4 Nano', provider: OPENAI },
  { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI },
  {
    id: GEMINI_PRO,
    name: 'Gemini 3.1 Pro Preview',
    provider: GOOGLE,
  },
  {
    id: CLAUDE_OPUS,
    name: 'Claude Opus 4.8',
    provider: ANTHROPIC,
  },
  {
    id: QWEN_THINKING,
    loaded: true,
    name: 'Qwen3 30B Thinking',
    provider: OLLAMA,
  },
];

const typedCatalog = (source: string = LIVE) => ({
  models: typedModels,
  source,
  version: 1,
});

describe('parseModelCatalog', () => {
  it('exposes the complete executable fallback catalog', () => {
    expect(Object.keys(CURATED_MODEL_DESCRIPTORS)).toStrictEqual(
      EXPECTED_CURATED_IDS,
    );
  });

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
    });
    expect(catalog.models.at(-1)).toMatchObject({
      id: QWEN_THINKING,
      loaded: true,
      provider: OLLAMA,
    });
  });

  it('preserves approved sponsored access metadata and strips secret fields', () => {
    const catalog = parseModelCatalog({
      models: [
        {
          api_key: 'do-not-forward',
          availability: 'sponsored',
          base_url: 'https://sponsor.invalid',
          endpoint: 'https://sponsor.invalid/v1',
          id: LUNA,
          name: 'GPT-5.6 Luna',
          provider: OPENAI,
          sponsored_quota: {
            limit: 10,
            remaining: 7,
            resets_at: '2026-07-18T12:00:00Z',
          },
        },
      ],
      source: LIVE,
      version: 1,
    });

    expect(catalog.models).toStrictEqual([
      {
        availability: 'sponsored',
        id: 'gpt-5.6-luna',
        name: 'GPT-5.6 Luna',
        provider: OPENAI,
        sponsored_quota: {
          limit: 10,
          remaining: 7,
          resets_at: '2026-07-18T12:00:00Z',
        },
      },
    ]);
  });

  it('drops malformed sponsored metadata without inferring sponsored access', () => {
    const catalog = parseModelCatalog({
      models: [
        {
          availability: 'sponsored-by-secret',
          id: GPT_MINI,
          name: GPT_MINI_NAME,
          provider: OPENAI,
          sponsored_quota: {
            limit: 2,
            remaining: 3,
            resets_at: 'not-a-reset',
          },
        },
      ],
      source: LIVE,
      version: 1,
    });

    expect(catalog.models).toStrictEqual([
      { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI },
    ]);
  });

  it('preserves an unknown Ollama loaded status', () => {
    const catalog = parseModelCatalog({
      models: [
        {
          id: 'llama3.2:latest',
          loaded: null,
          name: 'llama3.2:latest',
          provider: OLLAMA,
        },
      ],
      source: LIVE,
      version: 1,
    });

    expect(catalog.models[0]).toMatchObject({ loaded: null });
  });

  it('normalizes every curated legacy id with its immutable fallback descriptor', () => {
    const catalog = parseModelCatalog(EXPECTED_CURATED_IDS);

    expect(catalog.source).toBe(LIVE);
    expect(catalog.models).toStrictEqual(
      Object.values(CURATED_MODEL_DESCRIPTORS),
    );
  });

  it('uses inferred provider and raw id name for an unknown legacy id', () => {
    const unknown = 'acme/new-model';

    expect(parseModelCatalog([unknown]).models).toStrictEqual([
      descriptor(unknown, 'acme'),
    ]);
  });

  it.each([
    { models: typedModels, source: LIVE, version: 2 },
    { models: typedModels, source: 'weird', version: 1 },
    {
      models: [{ id: GPT_MINI, name: GPT_MINI_NAME }],
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
      models: [{ id: 'mystery-model', provider: OPENAI }],
      source: 'snapshot',
      version: 1,
    });

    expect(catalog).toStrictEqual({ models: [], source: 'error', version: 1 });
  });

  it('rejects the envelope when an entry has no string id', () => {
    const catalog = parseModelCatalog({
      models: [
        { name: 'No id', provider: OPENAI },
        { id: GPT_MINI, name: GPT_MINI_NAME, provider: OPENAI },
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

  it('treats an empty typed catalog as an error', () => {
    expect(
      parseModelCatalog({ models: [], source: LIVE, version: 1 }),
    ).toStrictEqual({
      models: [],
      source: 'error',
      version: 1,
    });
  });
});

describe('groupModelsByProvider', () => {
  it('groups by first-seen provider while preserving API order within providers', () => {
    const { models } = parseModelCatalog(typedCatalog());

    const groups = groupModelsByProvider(models);

    expect(groups.map(({ provider }) => provider)).toStrictEqual([
      OPENAI,
      GOOGLE,
      ANTHROPIC,
      OLLAMA,
    ]);
    expect(groups[0]?.models.map((model) => model.id)).toStrictEqual([
      GPT,
      GPT_NANO,
      GPT_MINI,
    ]);
  });

  it('is stable for models that share a provider', () => {
    const { models } = parseModelCatalog([GPT_TERRA, GPT_MINI]);

    const groups = groupModelsByProvider(models);

    expect(groups).toHaveLength(1);
    expect(groups[0]?.models.map((model) => model.id)).toStrictEqual([
      GPT_TERRA,
      GPT_MINI,
    ]);
  });
});

describe('model access predicates', () => {
  const openAiCredentials = new Set([OPENAI]);
  const noCredentials = new Set<string>();

  it('uses provider credentials for legacy descriptors with no availability', () => {
    const legacy = descriptor(GPT_MINI, OPENAI);

    expect(isModelAvailable(legacy, openAiCredentials)).toBe(true);
    expect(isModelAvailable(legacy, noCredentials)).toBe(false);
    expect(isSponsoredModel(legacy)).toBe(false);
  });

  it.each(['sponsored', 'both'] as const)(
    'allows %s models without provider credentials',
    (availability) => {
      const model = { ...descriptor(LUNA, OPENAI), availability };

      expect(isModelAvailable(model, noCredentials)).toBe(true);
      expect(isSponsoredModel(model)).toBe(true);
    },
  );

  it('requires BYOK credentials and blocks explicitly unavailable models', () => {
    expect(
      isModelAvailable(
        { ...descriptor(GPT_MINI, OPENAI), availability: 'byok' },
        noCredentials,
      ),
    ).toBe(false);
    expect(
      isModelAvailable(
        { ...descriptor(GPT_MINI, OPENAI), availability: 'byok' },
        openAiCredentials,
      ),
    ).toBe(true);
    expect(
      isModelAvailable(
        { ...descriptor(LUNA, OPENAI), availability: 'unavailable' },
        openAiCredentials,
      ),
    ).toBe(false);
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

/* eslint-enable camelcase -- end wire-contract fixtures. */

const LEGACY_MODEL_ALIASES: Readonly<Record<string, string>> = {
  'claude-haiku-4-5': 'claude-haiku-5-5',
  'claude-opus-4-8': 'claude-opus-5-5',
  'claude-sonnet-5': 'claude-sonnet-5-5',
  'gemini-3.1-flash-lite': 'gemini-3.5-flash-lite',
  'gpt-5.6-luna': 'gpt-6-luna',
  'gpt-5.6-sol': 'gpt-6.1-sol',
  'openrouter:x-ai/grok-4.6': 'openrouter:x-ai/grok-4.7',
};

export const normalizeModelId = (model: string): string => {
  const alias = Object.hasOwn(LEGACY_MODEL_ALIASES, model)
    ? LEGACY_MODEL_ALIASES[model]
    : undefined;
  return typeof alias === 'string' ? alias : model;
};

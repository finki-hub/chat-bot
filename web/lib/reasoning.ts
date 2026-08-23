const REASONING_PREFIXES = ['claude-', 'gemini-', 'gpt-5', 'openrouter:'];
const REASONING_MODEL_IDS = new Set([
  'qwen3:14b-q4_K_M',
  'qwen3:30b-a3b-thinking-2507-q4_K_M',
]);
const MANDATORY_REASONING_MODEL_IDS = new Set(['openrouter:z-ai/glm-5.3']);

export const isReasoningCapableModel = (model: string): boolean =>
  REASONING_MODEL_IDS.has(model) ||
  REASONING_PREFIXES.some((prefix) => model.startsWith(prefix));

export const isReasoningMandatoryModel = (model: string): boolean =>
  MANDATORY_REASONING_MODEL_IDS.has(model);

export const isReasoningEnabledForModel = (
  model: string,
  requested: boolean,
): boolean =>
  isReasoningMandatoryModel(model) ||
  (requested && isReasoningCapableModel(model));

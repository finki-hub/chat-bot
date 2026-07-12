const REASONING_PREFIXES = ['claude-', 'gemini-', 'gpt-5'];
const REASONING_MODEL_IDS = new Set([
  'qwen3:14b-q4_K_M',
  'qwen3:30b-a3b-thinking-2507-q4_K_M',
]);

export const isReasoningCapableModel = (model: string): boolean =>
  REASONING_MODEL_IDS.has(model) ||
  REASONING_PREFIXES.some((prefix) => model.startsWith(prefix));

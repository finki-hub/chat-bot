const REASONING_PREFIXES = [
  'claude-',
  'gemini-',
  'gpt-5',
  'gpt-6',
  'openrouter:',
];
const REASONING_MODEL_IDS = new Set([
  'gpt-6-astra',
  'qwen3:14b-q4_K_M',
  'qwen3:30b-a3b-thinking-2507-q4_K_M',
]);
const MANDATORY_REASONING_MODEL_IDS = new Set([
  'claude-fable-5-1',
  'claude-opus-5-5',
  'opencode-go:glm-5.3',
  'opencode-go:grok-4.7',
  'opencode-go:kimi-k3',
  'openrouter:qwen/qwen3.8-max-0902',
  'openrouter:x-ai/grok-4.7',
  'openrouter:z-ai/glm-5.3',
]);
// The Go catalog uses a closed allowlist: never infer capabilities from its provider
// prefix because upstream may expose models outside the curated, validated set.
const OPENCODE_GO_REASONING_MODEL_IDS = new Set([
  'opencode-go:claude-haiku-5-5',
  'opencode-go:deepseek-v4-pro',
  'opencode-go:deepseek-v4.1-flash',
  'opencode-go:glm-5.3',
  'opencode-go:gpt-6-luna',
  'opencode-go:grok-4.7',
  'opencode-go:kimi-k3',
  'opencode-go:minimax-m3',
  'opencode-go:qwen3.8-max',
]);

export const isReasoningCapableModel = (model: string): boolean =>
  REASONING_MODEL_IDS.has(model) ||
  OPENCODE_GO_REASONING_MODEL_IDS.has(model) ||
  REASONING_PREFIXES.some((prefix) => model.startsWith(prefix));

export const isReasoningMandatoryModel = (model: string): boolean =>
  MANDATORY_REASONING_MODEL_IDS.has(model);

export const isReasoningEnabledForModel = (
  model: string,
  requested: boolean,
): boolean =>
  isReasoningMandatoryModel(model) ||
  (requested && isReasoningCapableModel(model));

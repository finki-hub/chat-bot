// Models that expose an extended-thinking / reasoning mode. Anthropic (Claude),
// Google (Gemini), OpenAI (GPT-5.x), and DeepSeek-R1 surface reasoning; plain Qwen2
// (gpu-api) and the other local models do not, so the toggle stays disabled for them.
const REASONING_PREFIXES = ['claude-', 'gemini-', 'gpt-5', 'deepseek-r1'];

export const isReasoningCapableModel = (model: string): boolean =>
  REASONING_PREFIXES.some((prefix) => model.startsWith(prefix));

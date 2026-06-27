const REASONING_PREFIXES = ['claude-', 'gemini-', 'gpt-5', 'deepseek-r1'];

export const isReasoningCapableModel = (model: string): boolean =>
  REASONING_PREFIXES.some((prefix) => model.startsWith(prefix));

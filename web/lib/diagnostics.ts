import type { MessageDiagnostics } from '@/lib/api-types';

// Output tokens over the generation window (first byte → done), rounded to whole
// tokens/sec; null when the inputs to derive it are missing.
export const formatThroughput = (
  diagnostics: MessageDiagnostics,
): null | string => {
  const { serverTotalMs, serverTtftMs, tokens } = diagnostics;
  if (
    tokens === undefined ||
    typeof serverTotalMs !== 'number' ||
    typeof serverTtftMs !== 'number'
  ) {
    return null;
  }
  const generationMs = serverTotalMs - serverTtftMs;
  return generationMs <= 0 || tokens.output <= 0
    ? null
    : String(Math.round(tokens.output / (generationMs / 1_000)));
};

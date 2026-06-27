import { describe, expect, it } from 'vitest';

import type { MessageDiagnostics } from '@/lib/api-types';

import { formatThroughput } from '@/lib/diagnostics';

const diag = (overrides: MessageDiagnostics): MessageDiagnostics => overrides;

describe('formatThroughput', () => {
  it('derives whole tokens/sec from output tokens over the generation window', () => {
    expect(
      formatThroughput(
        diag({
          serverTotalMs: 1_200,
          serverTtftMs: 200,
          tokens: { input: 10, output: 60, total: 70 },
        }),
      ),
    ).toBe('60');
  });

  it('rounds the fractional rate to the nearest whole token/sec', () => {
    // 10 tokens over a 1.3s window → 7.69 tok/s, which must round up to 8.
    expect(
      formatThroughput(
        diag({
          serverTotalMs: 1_300,
          serverTtftMs: 0,
          tokens: { input: 1, output: 10, total: 11 },
        }),
      ),
    ).toBe('8');
  });

  it('returns null when tokens are missing', () => {
    expect(
      formatThroughput(diag({ serverTotalMs: 1_000, serverTtftMs: 100 })),
    ).toBeNull();
  });

  it('returns null when a timing value is missing', () => {
    expect(
      formatThroughput(
        diag({
          serverTotalMs: 1_000,
          tokens: { input: 1, output: 5, total: 6 },
        }),
      ),
    ).toBeNull();
  });

  it('returns null when the generation window is not positive', () => {
    expect(
      formatThroughput(
        diag({
          serverTotalMs: 500,
          serverTtftMs: 500,
          tokens: { input: 1, output: 5, total: 6 },
        }),
      ),
    ).toBeNull();
  });

  it('returns null when no output tokens were produced', () => {
    expect(
      formatThroughput(
        diag({
          serverTotalMs: 1_000,
          serverTtftMs: 100,
          tokens: { input: 5, output: 0, total: 5 },
        }),
      ),
    ).toBeNull();
  });
});

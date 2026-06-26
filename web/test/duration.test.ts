import { describe, expect, it } from 'vitest';

import { formatDuration } from '@/lib/duration';

describe('formatDuration', () => {
  it('renders sub-second values in milliseconds', () => {
    expect(formatDuration(0)).toBe('0ms');
    expect(formatDuration(850)).toBe('850ms');
    expect(formatDuration(999)).toBe('999ms');
  });

  it('rounds fractional milliseconds', () => {
    expect(formatDuration(123.4)).toBe('123ms');
    expect(formatDuration(123.6)).toBe('124ms');
  });

  it('renders one-second-and-above values in seconds with one decimal', () => {
    expect(formatDuration(1_000)).toBe('1.0s');
    expect(formatDuration(2_345)).toBe('2.3s');
    expect(formatDuration(12_000)).toBe('12.0s');
  });
});

import { describe, expect, it } from 'vitest';

describe('toolchain smoke', () => {
  it('has the jsdom environment available', () => {
    expect(document).toBeTypeOf('object');
    expect(document.createElement('div').tagName).toBe('DIV');
  });

  it('has the fake-indexeddb global polyfill', () => {
    expect(indexedDB).not.toBeTypeOf('undefined');
  });
});

import { beforeEach, expect, test, vi } from 'vitest';

import { ANON_USER_ID_KEY, getAnonUserId } from '@/lib/user';
import { createMemoryStorage } from '@/test/helpers/dom-stubs';

const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/u;

beforeEach(() => {
  vi.stubGlobal('localStorage', createMemoryStorage());
});

test('mints a UUID and persists it under the namespaced key', () => {
  const id = getAnonUserId();

  expect(id).toMatch(UUID_V4);
  expect(localStorage.getItem(ANON_USER_ID_KEY)).toBe(id);
});

test('returns the same id on subsequent calls', () => {
  const first = getAnonUserId();
  const second = getAnonUserId();

  expect(second).toBe(first);
});

test('reuses an id already present in localStorage', () => {
  localStorage.setItem(ANON_USER_ID_KEY, 'preexisting-id');

  expect(getAnonUserId()).toBe('preexisting-id');
});

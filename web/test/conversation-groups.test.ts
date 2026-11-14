/* eslint-disable unicorn/prefer-temporal -- the code under test takes a Date. */
import { describe, expect, it } from 'vitest';

import { groupConversationsByDate } from '@/lib/conversation-groups';
import { isTrustedLinkUrl } from '@/lib/trusted-links';

const NOW = new Date(2_026, 10, 14, 10, 0);
const at = (daysAgo: number, hour = 9): string =>
  new Date(2_026, 10, 14 - daysAgo, hour, 0).toISOString();
const row = (id: string, updatedAt?: string) => ({
  id,
  model: null,
  title: id,
  ...(updatedAt !== undefined && { updatedAt }),
});

const idsOf = (rows: ReadonlyArray<{ id: string }>): string[] =>
  rows.map((c) => c.id);

describe('groupConversationsByDate', () => {
  it('sorts conversations into calendar-day sections, newest first', () => {
    const groups = groupConversationsByDate(
      [
        row('new'),
        row('today', at(0)),
        row('late-yesterday', at(1, 23)),
        row('week', at(3)),
        row('month', at(20)),
        row('old', at(90)),
      ],
      NOW,
    );

    expect(
      groups?.map((group) => [group.label, idsOf(group.conversations)]),
    ).toStrictEqual([
      ['Денес', ['new', 'today']],
      ['Вчера', ['late-yesterday']],
      ['Последните 7 дена', ['week']],
      ['Последните 30 дена', ['month']],
      ['Постари', ['old']],
    ]);
  });

  it('stays flat when the API sends no timestamps', () => {
    expect(groupConversationsByDate([row('a'), row('b')], NOW)).toBeNull();
  });
});

describe('isTrustedLinkUrl', () => {
  it.each([
    ['https://finki.ukim.mk/mk/content/ceni', true],
    ['https://iknow.ukim.mk/', true],
    ['https://www.ukim.edu.mk/', true],
    ['https://chat.finki-hub.com/', true],
    // Only HTTPS opens directly.
    ['wss://finki.ukim.mk/', false],
    ['https://finki.ukim.mk.evil.com/', false],
    ['https://notfinki-hub.com/', false],
    ['not a url', false],
  ])('treats %s as trusted: %s', (url, trusted) => {
    expect(isTrustedLinkUrl(url)).toBe(trusted);
  });
});

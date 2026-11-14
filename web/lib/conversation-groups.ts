import type { ConversationRow } from '@/lib/conversation-types';

import { t, type TKey } from '@/lib/i18n';

const PERIODS = ['today', 'yesterday', 'week', 'month', 'older'] as const;

type Period = (typeof PERIODS)[number];

const PERIOD_LABELS: Record<Period, TKey> = {
  month: 'sidebar.month',
  older: 'sidebar.older',
  today: 'sidebar.today',
  week: 'sidebar.week',
  yesterday: 'sidebar.yesterday',
};

export type ConversationGroup = {
  readonly conversations: ConversationRow[];
  readonly key: Period;
  readonly label: string;
};

const DAY_MS = 86_400_000;

/* eslint-disable unicorn/prefer-temporal -- browser support is required here. */
const startOfDay = (date: Date): number =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

// Calendar days in the viewer's time zone, so "yesterday" means the previous date.
const periodOf = (updatedAt: string | undefined, now: Date): Period => {
  const time = updatedAt === undefined ? NaN : Date.parse(updatedAt);

  // A conversation started in this session has no server timestamp yet.
  if (Number.isNaN(time)) {
    return 'today';
  }

  const days = Math.round(
    (startOfDay(now) - startOfDay(new Date(time))) / DAY_MS,
  );

  if (days <= 0) {
    return 'today';
  }

  if (days === 1) {
    return 'yesterday';
  }

  if (days < 7) {
    return 'week';
  }

  return days < 30 ? 'month' : 'older';
};
/* eslint-enable unicorn/prefer-temporal -- end of the calendar-day helpers. */

// Date sections for the sidebar, newest first, keeping the server's order
// inside each. Without any timestamps (an older API) the list stays flat.
export const groupConversationsByDate = (
  conversations: readonly ConversationRow[],
  // eslint-disable-next-line unicorn/prefer-temporal -- browser support is required here.
  now: Date = new Date(),
): ConversationGroup[] | null => {
  if (conversations.every((row) => row.updatedAt === undefined)) {
    return null;
  }

  const byPeriod = new Map<Period, ConversationRow[]>();

  for (const row of conversations) {
    const period = periodOf(row.updatedAt, now);
    const group = byPeriod.get(period);

    if (group === undefined) {
      byPeriod.set(period, [row]);
    } else {
      group.push(row);
    }
  }

  return PERIODS.filter((period) => byPeriod.has(period)).map((period) => ({
    conversations: byPeriod.get(period) ?? [],
    key: period,
    label: t(PERIOD_LABELS[period]),
  }));
};

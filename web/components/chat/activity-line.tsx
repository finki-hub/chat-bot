'use client';

import { t } from '@/lib/i18n';
import { useElapsedSeconds } from '@/lib/use-elapsed-seconds';

// Elapsed time only appears once a wait is long enough to be worth counting.
const SHOW_ELAPSED_S = 3;
const SLOW_S = 15;
const LONGER_S = 45;

const formatElapsed = (seconds: number): string =>
  seconds < 60
    ? `${seconds} с`
    : `${Math.floor(seconds / 60)} мин ${String(seconds % 60).padStart(2, '0')} с`;

const ElapsedText = ({
  className = '',
  seconds,
}: {
  className?: string;
  seconds: number;
}) =>
  seconds < SHOW_ELAPSED_S ? null : (
    <span
      aria-hidden="true"
      className={`shrink-0 text-xs tabular-nums text-muted-foreground/70 motion-safe:animate-in motion-safe:fade-in-0 ${className}`}
      data-testid="elapsed"
    >
      {formatElapsed(seconds)}
    </span>
  );

export const Elapsed = ({ className }: { className?: string }) => (
  <ElapsedText
    className={className}
    seconds={useElapsedSeconds()}
  />
);

export const ActivityPulse = () => (
  <span
    aria-hidden="true"
    className="relative inline-flex size-4 shrink-0 items-center justify-center"
  >
    <span className="absolute inline-flex size-2.5 rounded-full bg-primary/50 motion-safe:animate-ping" />
    <span className="relative inline-flex size-2 rounded-full bg-primary" />
  </span>
);

// The single "still working" indicator for waits with nothing else moving on
// screen: a live label, elapsed time, and a reassurance once the wait drags on.
export const ActivityLine = ({ label }: { label: string }) => {
  const seconds = useElapsedSeconds();
  let hint: null | string = null;

  if (seconds >= LONGER_S) {
    hint = t('activity.longer');
  } else if (seconds >= SLOW_S) {
    hint = t('activity.slow');
  }

  return (
    <output
      aria-live="polite"
      className="flex flex-col gap-0.5 py-1 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-300"
      data-testid="activity-line"
    >
      <span className="flex items-center gap-2 text-sm">
        <ActivityPulse />
        <span className="text-shimmer min-w-0">{`${label}…`}</span>
        <ElapsedText seconds={seconds} />
      </span>
      {hint === null ? null : (
        <span
          className="pl-6 text-xs text-muted-foreground/80 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-500"
          data-testid="activity-hint"
        >
          {hint}
        </span>
      )}
    </output>
  );
};

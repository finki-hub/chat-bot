'use client';

import { useEffect, useState } from 'react';

const secondsSince = (start: number): number =>
  Math.max(0, Math.floor((performance.now() - start) / 1_000));

// Seconds since `startedAt` (a `performance.now()` timestamp), or since mount;
// remount (via `key`) to restart. Measured against a timestamp so throttled
// background-tab intervals still report real time.
export const useElapsedSeconds = (startedAt?: number): number => {
  const [elapsed, setElapsed] = useState(() =>
    startedAt === undefined ? 0 : secondsSince(startedAt),
  );

  useEffect(() => {
    const start = startedAt ?? performance.now();
    const id = setInterval(() => {
      setElapsed(secondsSince(start));
    }, 1_000);

    return () => {
      clearInterval(id);
    };
  }, [startedAt]);

  return elapsed;
};

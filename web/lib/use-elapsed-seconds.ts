'use client';

import { useEffect, useState } from 'react';

// Seconds since mount; remount (via `key`) to restart. Measured against a
// timestamp so throttled background-tab intervals still report real time.
export const useElapsedSeconds = (): number => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((performance.now() - start) / 1_000));
    }, 1_000);

    return () => {
      clearInterval(id);
    };
  }, []);

  return elapsed;
};

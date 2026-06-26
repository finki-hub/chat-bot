'use client';

import { useEffect, useState } from 'react';

import { formatDuration } from '@/lib/duration';

export type ElapsedTimerProps = {
  startedAt: number;
};

export const ElapsedTimer = ({ startedAt }: ElapsedTimerProps) => {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    setNow(Date.now());
    const id = setInterval(() => {
      setNow(Date.now());
    }, 200);

    return () => {
      clearInterval(id);
    };
  }, [startedAt]);

  return (
    <span
      className="text-xs tabular-nums text-muted-foreground/70"
      data-testid="elapsed-timer"
    >
      {formatDuration(Math.max(0, now - startedAt))}
    </span>
  );
};

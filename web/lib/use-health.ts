'use client';

import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';

const HEALTHY_INTERVAL_MS = 30_000;
const DOWN_INTERVAL_MS = 5_000;
const RECOVERY_CHECKS_TO_CLEAR = 2;

const fetchHealth = async (): Promise<true> => {
  const res = await fetch('/api/health');
  if (!res.ok) {
    throw new Error('unavailable');
  }

  return true;
};

export const useHealth = (): { unavailable: boolean } => {
  const [unavailable, setUnavailable] = useState(false);
  const recoveryChecks = useRef(0);
  const query = useQuery({
    queryFn: fetchHealth,
    queryKey: ['health'],
    refetchInterval: (q) =>
      unavailable || q.state.status === 'error'
        ? DOWN_INTERVAL_MS
        : HEALTHY_INTERVAL_MS,
    refetchOnWindowFocus: 'always',
    // One retry absorbs a transient blip before flagging the backend down.
    retry: 1,
    staleTime: HEALTHY_INTERVAL_MS,
  });

  useEffect(() => {
    if (query.fetchStatus !== 'idle') {
      return;
    }

    if (query.isError) {
      recoveryChecks.current = 0;
      setUnavailable(true);
      return;
    }

    if (!query.isSuccess) {
      return;
    }

    recoveryChecks.current += 1;
    if (recoveryChecks.current >= RECOVERY_CHECKS_TO_CLEAR) {
      setUnavailable(false);
    }
  }, [
    query.dataUpdatedAt,
    query.errorUpdatedAt,
    query.fetchStatus,
    query.isError,
    query.isSuccess,
  ]);

  return { unavailable };
};

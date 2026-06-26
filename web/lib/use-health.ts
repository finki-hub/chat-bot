'use client';

import { useQuery } from '@tanstack/react-query';

const HEALTHY_INTERVAL_MS = 30_000;
const DOWN_INTERVAL_MS = 5_000;

const fetchHealth = async (): Promise<true> => {
  const res = await fetch('/api/health');
  if (!res.ok) {
    throw new Error('unavailable');
  }

  return true;
};

export const useHealth = (): { unavailable: boolean } => {
  const query = useQuery({
    queryFn: fetchHealth,
    queryKey: ['health'],
    refetchInterval: (q) =>
      q.state.status === 'error' ? DOWN_INTERVAL_MS : HEALTHY_INTERVAL_MS,
    refetchOnWindowFocus: 'always',
    // One retry absorbs a transient blip before flagging the backend down.
    retry: 1,
    staleTime: HEALTHY_INTERVAL_MS,
  });

  // Optimistic: only flag unavailable after a check actually fails (post-retry).
  return { unavailable: query.isError };
};

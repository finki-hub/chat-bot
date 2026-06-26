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
    // One retry absorbs a single transient blip; the down-state still resolves
    // within ~one retry. Recovery is picked up by the faster down-interval and
    // the always-on focus refetch.
    retry: 1,
    staleTime: HEALTHY_INTERVAL_MS,
  });

  // Optimistic: only flag unavailable once a check has actually failed (after
  // the retry), so the chat is never blocked during the initial in-flight
  // check and a single blip can't lock the composer.
  return { unavailable: query.isError };
};

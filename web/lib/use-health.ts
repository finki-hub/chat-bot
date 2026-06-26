'use client';

import { useQuery } from '@tanstack/react-query';

const isOk = (value: unknown): boolean =>
  typeof value === 'object' &&
  value !== null &&
  (value as { ok?: unknown }).ok === true;

const fetchHealth = async (): Promise<boolean> => {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) {
      return false;
    }

    return isOk(await res.json());
  } catch {
    return false;
  }
};

export const useHealth = (): { unavailable: boolean } => {
  const query = useQuery({
    queryFn: fetchHealth,
    queryKey: ['health'],
    refetchInterval: 30_000,
    refetchOnWindowFocus: true,
    staleTime: 15_000,
  });

  // Optimistic: only flag as unavailable once a check has actually failed, so
  // the chat is never blocked during the initial in-flight check.
  return { unavailable: query.data === false };
};

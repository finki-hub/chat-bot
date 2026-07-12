'use client';

import { useQuery } from '@tanstack/react-query';

import type { ChatCredentialPublic } from '@/lib/api-types';

import { loadCredentials } from '@/components/shell/credential-settings-client';

export const CREDENTIALS_QUERY_KEY = ['credentials'] as const;
const EMPTY_CREDENTIALS: readonly ChatCredentialPublic[] = [];

export const useCredentials = () => {
  const query = useQuery({
    queryFn: ({ signal }) => loadCredentials(signal),
    queryKey: CREDENTIALS_QUERY_KEY,
    staleTime: 5 * 60 * 1_000,
  });
  return {
    credentials: query.data ?? EMPTY_CREDENTIALS,
    isError: query.isError || query.data === null,
    isLoading: query.isLoading,
    refetch: query.refetch,
  };
};

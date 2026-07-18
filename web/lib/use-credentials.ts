'use client';

import { useQuery } from '@tanstack/react-query';
import { useSession } from 'next-auth/react';

import type { ChatCredentialPublic } from '@/lib/api-types';

import { loadCredentials } from '@/components/shell/credential-settings-client';
import { getModelsSessionKey } from '@/lib/use-models';

export const CREDENTIALS_QUERY_KEY = ['credentials'] as const;
const EMPTY_CREDENTIALS: readonly ChatCredentialPublic[] = [];

export const useCredentials = () => {
  const { data: session, status } = useSession();
  const sessionKey = getModelsSessionKey(status, session);
  const query = useQuery({
    enabled: sessionKey !== null,
    queryFn: ({ signal }) => loadCredentials(signal),
    queryKey: [...CREDENTIALS_QUERY_KEY, sessionKey],
    staleTime: 5 * 60 * 1_000,
  });
  return {
    credentials: query.data ?? EMPTY_CREDENTIALS,
    isError: query.isError || query.data === null,
    isLoading: query.isLoading,
    refetch: query.refetch,
  };
};

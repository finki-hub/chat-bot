'use client';

import type { Session } from 'next-auth';

import { useQuery } from '@tanstack/react-query';
import { useSession } from 'next-auth/react';

import type { ModelCatalog } from '@/lib/api-types';

import { parseModelCatalog } from '@/lib/model-catalog';

const fetchCatalog = async (): Promise<ModelCatalog> => {
  const res = await fetch('/api/models');
  if (!res.ok) {
    throw new Error(`Failed to load models: ${res.status}`);
  }
  const data: unknown = await res.json();
  return parseModelCatalog(data);
};

export const MODELS_QUERY_KEY = ['models'] as const;

type SessionStatus = 'authenticated' | 'loading' | 'unauthenticated';

export const getModelsSessionKey = (
  status: SessionStatus,
  session: null | Session,
): null | string => {
  if (status === 'loading') {
    return null;
  }
  if (status === 'unauthenticated') {
    return 'anonymous';
  }

  const provider = session?.user?.provider;
  const providerSubject = session?.user?.providerSubject;
  if (provider !== undefined && providerSubject !== undefined) {
    return `${provider}:${providerSubject}`;
  }
  return 'authenticated:unknown';
};

export const useModels = () => {
  const { data: session, status } = useSession();
  const sessionKey = getModelsSessionKey(status, session);
  const query = useQuery({
    enabled: sessionKey !== null,
    queryFn: fetchCatalog,
    queryKey: [...MODELS_QUERY_KEY, sessionKey],
    staleTime: 5 * 60 * 1_000,
  });
  return {
    isError: query.isError || query.data?.source === 'error',
    isLoading: query.isLoading,
    models: query.data?.models ?? [],
    refetch: query.refetch,
    source: query.data?.source,
  };
};

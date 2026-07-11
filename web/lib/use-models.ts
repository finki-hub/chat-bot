'use client';

import { useQuery } from '@tanstack/react-query';

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

export const useModels = () => {
  const query = useQuery({
    queryFn: fetchCatalog,
    queryKey: ['models'],
    staleTime: 5 * 60 * 1_000,
  });
  return {
    isError: query.isError || query.data?.source === 'error',
    isLoading: query.isLoading,
    models: query.data?.models ?? [],
    source: query.data?.source,
  };
};

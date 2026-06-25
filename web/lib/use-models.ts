'use client';

import { useQuery } from '@tanstack/react-query';

import type { ModelId } from '@/lib/api-types';

const fetchModels = async (): Promise<ModelId[]> => {
  const res = await fetch('/api/models');
  if (!res.ok) {
    throw new Error(`Failed to load models: ${res.status}`);
  }
  const data: unknown = await res.json();
  return Array.isArray(data) ? (data as ModelId[]) : [];
};

export const useModels = () => {
  const query = useQuery({
    queryFn: fetchModels,
    queryKey: ['models'],
    staleTime: 5 * 60 * 1_000,
  });
  return {
    data: query.data,
    isError: query.isError,
    isLoading: query.isLoading,
  };
};

const providerOf = (id: string): string => {
  const slash = id.indexOf('/');
  if (slash > 0) {
    return id.slice(0, slash);
  }
  const dash = id.indexOf('-');
  if (dash > 0) {
    return id.slice(0, dash);
  }
  return id.length > 0 ? id : 'other';
};

export type ModelGroup = {
  models: string[];
  provider: string;
};

const collator = new Intl.Collator();

export const groupModelsByProvider = (ids: string[]): ModelGroup[] => {
  const map = new Map<string, string[]>();
  for (const id of ids) {
    const provider = providerOf(id);
    const bucket = map.get(provider);
    if (bucket) {
      bucket.push(id);
    } else {
      map.set(provider, [id]);
    }
  }
  return [...map]
    .map(([provider, models]) => ({
      models: models.toSorted((a, b) => collator.compare(a, b)),
      provider,
    }))
    .sort((a, b) => collator.compare(a.provider, b.provider));
};

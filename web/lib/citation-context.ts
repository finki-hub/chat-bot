import { createContext, useContext } from 'react';

import type { RetrievedSource } from '@/lib/api-types';

export type CitationState = {
  // The card a citation last pointed at, highlighted until the panel closes.
  readonly focusedRef: null | number;
  readonly openSource: (ref: number) => void;
  readonly setSourcesOpen: (open: boolean) => void;
  readonly sources: readonly RetrievedSource[];
  readonly sourcesOpen: boolean;
};

export const CitationContext = createContext<CitationState | null>(null);

export const useCitations = (): CitationState | null =>
  useContext(CitationContext);

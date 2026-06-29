import { useEffect, useMemo, useState } from 'react';

import {
  isSearchStage,
  SEARCH_STAGES,
  type SearchStage,
} from '@/lib/search-status';

export type UseSearchStageArgs = {
  pending?: boolean;
  reasoningActive?: boolean;
  stage?: string;
  statusActive?: boolean;
  text: null | string;
};

// Returns the pipeline stages to reveal, in canonical order, with the last entry
// being the in-progress one. Retrieval stages are tracked by the furthest reached
// (reaching one implies the earlier ones ran, even when several status events land
// in a single render), while generation is tracked separately so a reasoning-only
// turn shows just `generate` and never fabricates retrieval steps that never ran.
export const useSearchStage = ({
  pending,
  reasoningActive,
  stage,
  statusActive,
  text,
}: UseSearchStageArgs): SearchStage[] => {
  const [maxRetrieval, setMaxRetrieval] = useState(-1);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    if (stage !== undefined && isSearchStage(stage)) {
      if (stage === 'generate') {
        setGenerating(true);
      } else {
        const index = SEARCH_STAGES.indexOf(stage);
        setMaxRetrieval((prev) => Math.max(index, prev));
        // The pipeline path keeps the terminal `context` status active through
        // generation (it sends no reset). Once the model starts thinking,
        // generation has begun — advance the stepper to `generate`.
        if (stage === 'context' && reasoningActive === true) {
          setGenerating(true);
        }
      }
    } else if (pending === true && statusActive !== true && text === null) {
      // A reset cleared the status; the model is now generating the answer.
      setGenerating(true);
    }
  }, [stage, pending, reasoningActive, statusActive, text]);

  return useMemo<SearchStage[]>(() => {
    const stages: SearchStage[] =
      maxRetrieval >= 0 ? SEARCH_STAGES.slice(0, maxRetrieval + 1) : [];
    if (generating) {
      stages.push('generate');
    }
    return stages;
  }, [maxRetrieval, generating]);
};

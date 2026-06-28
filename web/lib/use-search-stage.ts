import { useEffect, useState } from 'react';

import {
  isSearchStage,
  type SearchStage,
  searchStageIndex,
} from '@/lib/search-status';

export type UseSearchStageArgs = {
  pending?: boolean;
  stage?: string;
  text: null | string;
};

export const useSearchStage = ({
  pending,
  stage,
  text,
}: UseSearchStageArgs): SearchStage | undefined => {
  const [maxStage, setMaxStage] = useState<SearchStage | undefined>(undefined);

  useEffect(() => {
    let next: SearchStage | undefined;
    if (stage && isSearchStage(stage)) {
      next = stage;
    } else if (pending && !stage && text === null) {
      // A reset cleared the status; the model is now generating the answer.
      next = 'generate';
    }

    if (next === undefined) {
      return;
    }

    const resolved = next;
    setMaxStage((prev) =>
      prev === undefined || searchStageIndex(resolved) > searchStageIndex(prev)
        ? resolved
        : prev,
    );
  }, [stage, pending, text]);

  return maxStage;
};

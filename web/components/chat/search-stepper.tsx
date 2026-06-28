import { Check, Loader2 } from 'lucide-react';
import { type ReactNode } from 'react';

import {
  SEARCH_STAGES,
  type SearchStage,
  searchStageIndex,
  searchStageLabel,
} from '@/lib/search-status';

export type SearchStepperProps = {
  activeStage: SearchStage;
};

export const SearchStepper = ({ activeStage }: SearchStepperProps) => {
  const activeIndex = searchStageIndex(activeStage);

  return (
    <div
      aria-live="polite"
      className="flex flex-col gap-1"
      data-testid="search-stepper"
    >
      {SEARCH_STAGES.map((stage, index) => {
        const completed = index < activeIndex;
        const active = index === activeIndex;

        let icon: ReactNode;
        if (completed) {
          icon = <Check className="size-3.5" />;
        } else if (active) {
          icon = <Loader2 className="size-4 animate-spin" />;
        } else {
          icon = <span className="size-2 rounded-full bg-current opacity-40" />;
        }

        let colorClass = 'text-muted-foreground/70';
        if (active) {
          colorClass = 'font-medium text-foreground';
        } else if (completed) {
          colorClass = 'text-muted-foreground';
        }

        return (
          <div
            className={`flex items-center gap-2 text-sm transition-colors duration-200 ${colorClass}`}
            key={stage}
          >
            <span className="inline-flex size-4 items-center justify-center">
              {icon}
            </span>
            <span>{searchStageLabel(stage)}</span>
          </div>
        );
      })}
    </div>
  );
};

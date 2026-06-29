import { Check, Loader2 } from 'lucide-react';
import { type ReactNode } from 'react';

import { type SearchStage, searchStageLabel } from '@/lib/search-status';

export type SearchStepperProps = {
  /** Reached stages in canonical order; the last entry is the in-progress one. */
  stages: SearchStage[];
};

export const SearchStepper = ({ stages }: SearchStepperProps) => {
  const activeIndex = stages.length - 1;

  return (
    <div
      aria-live="polite"
      className="flex flex-col gap-1"
      data-testid="search-stepper"
    >
      {stages.map((stage, index) => {
        const active = index === activeIndex;

        const icon: ReactNode = active ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <Check className="size-3.5" />
        );

        const colorClass = active
          ? 'font-medium text-foreground'
          : 'text-muted-foreground';

        return (
          <div
            className={`flex items-center gap-2 text-sm transition-colors duration-200 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-top-1 motion-safe:duration-300 ${colorClass}`}
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

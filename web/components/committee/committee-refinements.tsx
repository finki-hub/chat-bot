import { Ban, Pin, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { t } from '@/lib/i18n';

export type CommitteeRefinements = {
  excluded: string[];
  pinned: string[];
};

export const MAX_PINNED_MEMBERS = 2;

export const PersonActions = ({
  canPin,
  isPinned,
  name,
  onChooseMentor,
  onExclude,
  onTogglePin,
}: {
  canPin: boolean;
  isPinned: boolean;
  name: string;
  onChooseMentor?: (name: string) => void;
  onExclude: (name: string) => void;
  onTogglePin?: (name: string) => void;
}) => (
  <div className="mt-3 flex flex-wrap gap-2">
    {onChooseMentor ? (
      <Button
        aria-label={`${t('committee.action.chooseMentor')}: ${name}`}
        className="pointer-coarse:min-h-11"
        onClick={() => {
          onChooseMentor(name);
        }}
        size="sm"
        type="button"
        variant="outline"
      >
        <Pin className="size-3.5" />
        {t('committee.action.chooseMentor')}
      </Button>
    ) : null}
    {onTogglePin ? (
      <Button
        aria-label={`${t('committee.action.keep')}: ${name}`}
        aria-pressed={isPinned}
        className="pointer-coarse:min-h-11"
        disabled={!canPin && !isPinned}
        onClick={() => {
          onTogglePin(name);
        }}
        size="sm"
        type="button"
        variant={isPinned ? 'secondary' : 'outline'}
      >
        <Pin className="size-3.5" />
        {isPinned ? t('committee.action.kept') : t('committee.action.keep')}
      </Button>
    ) : null}
    <Button
      aria-label={`${t('committee.action.exclude')}: ${name}`}
      className="text-muted-foreground pointer-coarse:min-h-11"
      onClick={() => {
        onExclude(name);
      }}
      size="sm"
      type="button"
      variant="ghost"
    >
      <Ban className="size-3.5" />
      {t('committee.action.exclude')}
    </Button>
  </div>
);

const RefinementChip = ({
  label,
  name,
  onRemove,
}: {
  label: string;
  name: string;
  onRemove: (name: string) => void;
}) => (
  <li className="inline-flex items-center gap-1 rounded-full bg-secondary py-0.5 pl-2.5 pr-1 text-xs font-medium text-secondary-foreground">
    {label}: {name}
    <button
      aria-label={`${t('committee.refine.remove')}: ${name}`}
      className="inline-flex size-5 items-center justify-center rounded-full hover:bg-secondary-foreground/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:size-7"
      onClick={() => {
        onRemove(name);
      }}
      type="button"
    >
      <X className="size-3" />
    </button>
  </li>
);

export const RefinementBar = ({
  onRemove,
  refinements,
}: {
  onRemove: (name: string) => void;
  refinements: CommitteeRefinements;
}) => {
  if (refinements.pinned.length === 0 && refinements.excluded.length === 0) {
    return null;
  }

  return (
    <section className="mb-4 rounded-xl border border-border bg-muted/30 p-3">
      <h2 className="mb-2 text-xs font-medium text-muted-foreground">
        {t('committee.refine.title')}
      </h2>
      <ul className="flex flex-wrap gap-2">
        {refinements.pinned.map((name) => (
          <RefinementChip
            key={`pinned-${name}`}
            label={t('committee.action.kept')}
            name={name}
            onRemove={onRemove}
          />
        ))}
        {refinements.excluded.map((name) => (
          <RefinementChip
            key={`excluded-${name}`}
            label={t('committee.refine.excluded')}
            name={name}
            onRemove={onRemove}
          />
        ))}
      </ul>
    </section>
  );
};

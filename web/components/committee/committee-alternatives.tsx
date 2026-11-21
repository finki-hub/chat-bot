import type { CommitteeAlternative, Professor } from '@/lib/committee-types';

import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

const findProfessor = (
  name: string,
  professors: Professor[],
): Professor | undefined =>
  professors.find((professor) => professor.name === name);

const AlternativeRow = ({
  alternative,
  index,
  isRecommended,
  professors,
}: {
  alternative: CommitteeAlternative;
  index: number;
  isRecommended: boolean;
  professors: Professor[];
}) => {
  const mentorProfessor = alternative.mentor
    ? findProfessor(alternative.mentor, professors)
    : undefined;

  return (
    <div
      className={cn(
        'rounded-lg border p-3',
        isRecommended
          ? 'border-primary/40 bg-primary/5'
          : 'border-border bg-card',
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span
          className={cn(
            'inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium',
            isRecommended
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground',
          )}
        >
          {isRecommended
            ? t('committee.result.alternativeRecommended')
            : t('committee.result.alternativeConsider')}
        </span>
        <span className="text-xs text-muted-foreground">#{index + 1}</span>
      </div>
      <div className="mt-2 space-y-1">
        {alternative.mentor ? (
          <p className="text-sm font-medium text-foreground">
            {mentorProfessor?.title ? `${mentorProfessor.title} ` : null}
            {alternative.mentor}
          </p>
        ) : null}
        {alternative.members.length > 0 ? (
          <p className="text-sm text-muted-foreground">
            {alternative.members.join(', ')}
          </p>
        ) : null}
      </div>
    </div>
  );
};

export const AlternativesSection = ({
  alternatives,
  professors,
}: {
  alternatives: CommitteeAlternative[];
  professors: Professor[];
}) => {
  if (alternatives.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-foreground">
        {t('committee.result.alternatives')}
      </h3>
      <div className="grid gap-3 sm:grid-cols-2">
        {alternatives.map((alternative, index) => (
          <AlternativeRow
            alternative={alternative}
            index={index}
            // The API includes the selected committee first.
            isRecommended={index === 0}
            key={`${alternative.mentor ?? 'none'}-${alternative.members.join('-')}`}
            professors={professors}
          />
        ))}
      </div>
    </div>
  );
};

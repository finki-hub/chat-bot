import type { CommitteeAlternative, Professor } from '@/lib/committee-types';

import { formatDefenseCount } from '@/lib/committee-format';
import { t } from '@/lib/i18n';

const findProfessor = (
  name: string,
  professors: Professor[],
): Professor | undefined =>
  professors.find((professor) => professor.name === name);

const AlternativeRow = ({
  alternative,
  professors,
}: {
  alternative: CommitteeAlternative;
  professors: Professor[];
}) => {
  const mentorProfessor = alternative.mentor
    ? findProfessor(alternative.mentor, professors)
    : undefined;
  const defenses = alternative.supporting_diploma_ids.length;

  return (
    <li className="rounded-lg border border-border bg-card p-3">
      {alternative.mentor ? (
        <>
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {t('committee.result.mentorLabel')}
            {mentorProfessor?.title ? ` · ${mentorProfessor.title}` : null}
          </p>
          <p className="text-sm font-semibold text-foreground">
            {alternative.mentor}
          </p>
        </>
      ) : null}
      {alternative.members.length > 0 ? (
        <p className="mt-1 text-sm text-muted-foreground">
          {alternative.members.join(', ')}
        </p>
      ) : null}
      {defenses > 0 ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {formatDefenseCount(defenses)}
        </p>
      ) : null}
    </li>
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
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-foreground">
          {t('committee.result.alternatives')}
        </h3>
        <p className="text-sm text-muted-foreground">
          {t('committee.result.alternativeConsider')}
        </p>
      </div>
      <ul className="grid gap-3 sm:grid-cols-2">
        {alternatives.map((alternative) => (
          <AlternativeRow
            alternative={alternative}
            key={`${alternative.mentor ?? 'none'}-${alternative.members.join('-')}`}
            professors={professors}
          />
        ))}
      </ul>
    </section>
  );
};

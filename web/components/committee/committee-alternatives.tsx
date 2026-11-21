import type { CommitteeAlternative, Professor } from '@/lib/committee-types';

import { Button } from '@/components/ui/button';
import { formatDefenseCount } from '@/lib/committee-format';
import { t } from '@/lib/i18n';

const findProfessor = (
  name: string,
  professors: Professor[],
): Professor | undefined =>
  professors.find((professor) => professor.name === name);

const AlternativeRow = ({
  alternative,
  chooseLabel,
  onChoose,
  professors,
  selectedMentor,
}: {
  alternative: CommitteeAlternative;
  chooseLabel: string;
  onChoose: (alternative: CommitteeAlternative) => void;
  professors: Professor[];
  selectedMentor: string;
}) => {
  const showMentor =
    alternative.mentor !== null && alternative.mentor !== selectedMentor;
  const mentorProfessor = alternative.mentor
    ? findProfessor(alternative.mentor, professors)
    : undefined;
  const defenses = alternative.supporting_diploma_ids.length;
  const isPaperBased = alternative.mentor_basis === 'papers';

  return (
    <li className="flex flex-col rounded-lg border border-border bg-card p-3">
      {showMentor ? (
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
      {isPaperBased ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {t('committee.result.paperBasis')}
        </p>
      ) : null}
      {!isPaperBased && defenses > 0 ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {formatDefenseCount(defenses)}
        </p>
      ) : null}
      <Button
        aria-label={`${chooseLabel}: ${
          showMentor ? alternative.mentor : alternative.members.join(', ')
        }`}
        className="mt-3 self-start pointer-coarse:min-h-11"
        onClick={() => {
          onChoose(alternative);
        }}
        size="sm"
        type="button"
        variant="outline"
      >
        {chooseLabel}
      </Button>
    </li>
  );
};

export const AlternativesSection = ({
  alternatives,
  mentorIsGiven,
  onChoose,
  professors,
  selectedMentor,
}: {
  alternatives: CommitteeAlternative[];
  mentorIsGiven: boolean;
  onChoose: (alternative: CommitteeAlternative) => void;
  professors: Professor[];
  selectedMentor: string;
}) => {
  if (alternatives.length === 0) return null;

  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-foreground">
          {mentorIsGiven
            ? t('committee.result.alternatives')
            : t('committee.result.shortlist')}
        </h3>
        <p className="text-sm text-muted-foreground">
          {mentorIsGiven
            ? t('committee.result.alternativeConsider')
            : t('committee.result.shortlistHint')}
        </p>
      </div>
      <ul className="grid gap-3 sm:grid-cols-2">
        {alternatives.map((alternative) => (
          <AlternativeRow
            alternative={alternative}
            chooseLabel={
              mentorIsGiven
                ? t('committee.action.useMembers')
                : t('committee.action.chooseMentor')
            }
            key={`${alternative.mentor ?? 'none'}-${alternative.members.join('-')}`}
            onChoose={onChoose}
            professors={professors}
            selectedMentor={selectedMentor}
          />
        ))}
      </ul>
    </section>
  );
};

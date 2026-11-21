import type { ReactNode } from 'react';

import { Check } from 'lucide-react';

import type {
  CommitteePerson,
  CommitteeResponse,
  Professor,
} from '@/lib/committee-types';

import { t } from '@/lib/i18n';

const NAME_PARTS = /\s+/u;

const toInitials = (name: string): string => {
  const parts = name.trim().split(NAME_PARTS);
  const first = parts[0]?.[0] ?? '';
  const last = parts.at(-1)?.[0] ?? '';
  return `${first}${last}`.toUpperCase();
};

const findProfessor = (
  name: string,
  professors: Professor[],
): Professor | undefined =>
  professors.find((professor) => professor.name === name);

const ProfessorAvatar = ({
  name,
  photo,
}: {
  name: string;
  photo: null | string;
}) => (
  <span className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-full bg-muted text-sm font-semibold text-muted-foreground ring-1 ring-border">
    {photo ? (
      <img
        alt=""
        className="size-full object-cover"
        src={photo}
      />
    ) : (
      toInitials(name)
    )}
  </span>
);

const ProfessorLinks = ({
  professor,
}: {
  professor: Professor | undefined;
}) => {
  if (!professor) return null;

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
      {professor.profile ? (
        <a
          className="inline-flex items-center gap-1 text-primary underline-offset-2 hover:underline"
          href={professor.profile}
          rel="noopener noreferrer"
          target="_blank"
        >
          {t('committee.profile')}
        </a>
      ) : null}
      {professor.consultations ? (
        <span className="text-muted-foreground">
          {t('committee.consultations')}: {professor.consultations}
        </span>
      ) : null}
      {professor.cabinet ? (
        <span className="text-muted-foreground">
          {t('committee.cabinet')}: {professor.cabinet}
        </span>
      ) : null}
    </div>
  );
};

const WhyBullet = ({ children }: { children: ReactNode }) => (
  <li className="flex items-start gap-2 text-sm text-muted-foreground">
    <Check className="mt-0.5 size-4 shrink-0 text-success" />
    <span className="flex-1">{children}</span>
  </li>
);

const WhySection = ({
  evidence,
  person,
}: {
  evidence: CommitteeResponse['evidence'];
  person: CommitteePerson;
}) => {
  const bullets: ReactNode[] = [];

  if (person.supporting_diploma_ids.length > 0) {
    bullets.push(
      <WhyBullet key="diploma-count">
        {t('committee.why.similarDefenses')}:{' '}
        {person.supporting_diploma_ids.length}
      </WhyBullet>,
    );
  }

  for (const title of person.supporting_paper_titles.slice(0, 3)) {
    bullets.push(
      <WhyBullet key={`paper-${title}`}>
        {t('committee.why.supportingPapers')}: {title}
      </WhyBullet>,
    );
  }

  const matchingDiplomas = evidence.similar_diplomas.filter((diploma) =>
    [diploma.mentor, ...diploma.members].includes(person.name),
  );

  for (const diploma of matchingDiplomas.slice(0, 3)) {
    bullets.push(
      <WhyBullet key={`diploma-${diploma.id}`}>
        {diploma.title}
        {diploma.mentor
          ? ` · ${t('committee.why.mentor')}: ${diploma.mentor}`
          : null}
        {diploma.members.length > 0
          ? ` · ${t('committee.why.members')}: ${diploma.members.join(', ')}`
          : null}
      </WhyBullet>,
    );
  }

  if (bullets.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        {t('committee.result.noMatches')}
      </p>
    );
  }

  return <ul className="space-y-1.5">{bullets}</ul>;
};

export const PersonCard = ({
  evidence,
  label,
  person,
  professors,
}: {
  evidence: CommitteeResponse['evidence'];
  label: string;
  person: CommitteePerson;
  professors: Professor[];
}) => {
  const professor = findProfessor(person.name, professors);

  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <ProfessorAvatar
          name={person.name}
          photo={professor?.photo ?? null}
        />
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          <p className="truncate text-base font-semibold text-foreground">
            {person.name}
          </p>
          {professor && (professor.title || professor.position) ? (
            <p className="text-sm text-muted-foreground">
              {[professor.title, professor.position]
                .filter(Boolean)
                .join(' · ')}
            </p>
          ) : null}
          <div className="mt-2">
            <ProfessorLinks professor={professor} />
          </div>
        </div>
      </div>
      <div className="mt-4 border-t border-border/60 pt-3">
        <p className="mb-2 text-xs font-medium text-foreground">
          {t('committee.result.why')}
        </p>
        <WhySection
          evidence={evidence}
          person={person}
        />
      </div>
    </div>
  );
};

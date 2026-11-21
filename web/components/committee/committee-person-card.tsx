import type { ReactNode } from 'react';

import { BookOpen, ChevronDown, FileText } from 'lucide-react';

import type { CommitteePerson, Professor } from '@/lib/committee-types';

import { formatDefenseCount, formatPaperCount } from '@/lib/committee-format';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

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
  <span className="flex size-11 shrink-0 items-center justify-center overflow-hidden rounded-full bg-muted text-sm font-semibold text-muted-foreground ring-1 ring-border">
    {photo ? (
      <img
        alt=""
        className="size-full object-cover"
        decoding="async"
        loading="lazy"
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
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm pointer-coarse:gap-y-0">
      {professor.profile ? (
        <a
          className="inline-flex items-center gap-1 text-primary underline-offset-2 hover:underline pointer-coarse:min-h-11"
          href={professor.profile}
          rel="noopener noreferrer"
          target="_blank"
        >
          {t('committee.profile')}
        </a>
      ) : null}
      {professor.consultations ? (
        <a
          className="inline-flex items-center gap-1 text-primary underline-offset-2 hover:underline pointer-coarse:min-h-11"
          href={professor.consultations}
          rel="noopener noreferrer"
          target="_blank"
        >
          {t('committee.consultations')}
        </a>
      ) : null}
      {professor.cabinet ? (
        <span className="text-muted-foreground">
          {t('committee.cabinet')}: {professor.cabinet}
        </span>
      ) : null}
    </div>
  );
};

const PersonSignals = ({ person }: { person: CommitteePerson }) => {
  const defenses = person.supporting_diploma_ids.length;
  const papers = person.supporting_paper_titles;

  if (defenses === 0 && papers.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        {t('committee.result.noMatches')}
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {defenses > 0 ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <BookOpen
            aria-hidden="true"
            className="size-4 shrink-0 text-primary"
          />
          {formatDefenseCount(defenses)}
        </p>
      ) : null}
      {papers.length > 0 ? (
        <details className="group text-sm text-muted-foreground">
          <summary className="flex cursor-pointer list-none items-center gap-2 rounded-sm hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:min-h-11 [&::-webkit-details-marker]:hidden">
            <FileText
              aria-hidden="true"
              className="size-4 shrink-0 text-primary"
            />
            {formatPaperCount(papers.length)}
            <ChevronDown
              aria-hidden="true"
              className="size-4 shrink-0 transition-transform group-open:rotate-180 motion-reduce:transition-none"
            />
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-10">
            {papers.slice(0, 3).map((title) => (
              <li key={title}>
                <a
                  className="underline-offset-2 hover:text-foreground hover:underline"
                  href={`https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`}
                  rel="noopener noreferrer"
                  target="_blank"
                >
                  {title}
                </a>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
};

export const PersonCard = ({
  actions,
  featured = false,
  label,
  person,
  professors,
}: {
  actions?: ReactNode;
  featured?: boolean;
  label: string;
  person: CommitteePerson;
  professors: Professor[];
}) => {
  const professor = findProfessor(person.name, professors);

  return (
    <div
      className={cn(
        'rounded-xl border p-4',
        featured ? 'border-primary/40 bg-primary/5' : 'border-border bg-card',
      )}
    >
      <div className="flex items-start gap-3">
        <ProfessorAvatar
          name={person.name}
          photo={professor?.photo ?? null}
        />
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          <p className="text-base font-semibold text-pretty wrap-anywhere text-foreground">
            {person.name}
          </p>
          {professor && (professor.title || professor.position) ? (
            <p className="text-sm text-muted-foreground">
              {[professor.title, professor.position]
                .filter(Boolean)
                .join(' · ')}
            </p>
          ) : null}
          <div className="mt-1.5">
            <ProfessorLinks professor={professor} />
          </div>
        </div>
      </div>
      <div className="mt-3 border-t border-border/60 pt-3">
        <PersonSignals person={person} />
        {actions}
      </div>
    </div>
  );
};

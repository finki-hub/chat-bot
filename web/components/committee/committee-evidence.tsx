import { BookOpen } from 'lucide-react';

import type { SimilarDiploma } from '@/lib/committee-types';

import { t } from '@/lib/i18n';

export const EvidenceBlock = ({ diplomas }: { diplomas: SimilarDiploma[] }) => {
  if (diplomas.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-muted/30 p-4">
        <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-foreground">
          <BookOpen className="size-4" />
          {t('committee.evidence.title')}
        </h3>
        <p className="text-sm text-muted-foreground">
          {t('committee.evidence.empty')}
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-muted/30 p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-foreground">
        <BookOpen className="size-4" />
        {t('committee.evidence.title')}
      </h3>
      <ul className="space-y-3">
        {diplomas.slice(0, 5).map((diploma) => (
          <li
            className="rounded-lg border border-border bg-card p-3"
            key={diploma.id}
          >
            <p className="text-sm font-medium text-foreground">
              {diploma.title}
            </p>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
              {diploma.mentor ? (
                <span>
                  {t('committee.evidence.mentor')}: {diploma.mentor}
                </span>
              ) : null}
              {diploma.members.length > 0 ? (
                <span>
                  {t('committee.evidence.committee')}:{' '}
                  {diploma.members.join(', ')}
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

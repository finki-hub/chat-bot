import type { Metadata } from 'next';

import { CommitteePlanner } from '@/components/committee/committee-planner';
import { t } from '@/lib/i18n';

export const metadata: Metadata = {
  title: t('committee.pageTitle'),
};

export default function CommitteePage() {
  return (
    <main
      className="min-h-dvh bg-background text-foreground"
      id="main-content"
      tabIndex={-1}
    >
      <CommitteePlanner />
    </main>
  );
}

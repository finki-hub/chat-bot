'use client';

import { t } from '@/lib/i18n';

type ErrorPageProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

const ErrorPage = ({ reset }: ErrorPageProps) => (
  <main
    className="flex h-dvh w-full flex-col items-center justify-center gap-4 p-6 text-center"
    id="main-content"
    tabIndex={-1}
  >
    <div className="space-y-2">
      <h1 className="text-xl font-bold tracking-tight">{t('error.title')}</h1>
      <p className="max-w-md text-muted-foreground">{t('error.description')}</p>
    </div>
    <button
      className="rounded-md border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:min-h-11"
      onClick={reset}
      type="button"
    >
      {t('error.retry')}
    </button>
  </main>
);

export default ErrorPage;

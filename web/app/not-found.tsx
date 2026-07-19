import Link from 'next/link';

import { t } from '@/lib/i18n';

const NotFound = () => (
  <main
    className="flex h-dvh w-full flex-col items-center justify-center gap-4 p-6 text-center"
    id="main-content"
    tabIndex={-1}
  >
    <div className="space-y-2">
      <h1 className="text-xl font-bold tracking-tight">
        {t('notFound.title')}
      </h1>
      <p className="max-w-md text-muted-foreground">
        {t('notFound.description')}
      </p>
    </div>
    <Link
      className="rounded-md border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:min-h-11"
      href="/"
    >
      {t('notFound.home')}
    </Link>
  </main>
);

export default NotFound;

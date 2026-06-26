import { TriangleAlert } from 'lucide-react';

import { t } from '@/lib/i18n';

export const ServiceBanner = () => (
  <output
    aria-live="polite"
    className="flex shrink-0 items-center justify-center gap-2 border-b border-amber-500/20 bg-amber-500/10 px-4 py-2 text-center text-sm font-medium text-amber-700 dark:text-amber-400"
    data-testid="service-banner"
  >
    <TriangleAlert
      aria-hidden="true"
      className="size-4 shrink-0"
    />
    {t('service.unavailable')}
  </output>
);

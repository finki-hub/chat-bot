'use client';

import { ArrowRight } from 'lucide-react';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { Button } from '@/components/ui/button';
import { t } from '@/lib/i18n';

const DATE_FORMAT = new Intl.DateTimeFormat('mk-MK', {
  day: 'numeric',
  month: 'long',
  year: 'numeric',
});

const formatUpdated = (value: string | undefined): null | string => {
  if (value === undefined) {
    return null;
  }

  // eslint-disable-next-line unicorn/prefer-temporal -- browser support is required here.
  const date = new Date(value);

  return Number.isNaN(date.getTime()) ? null : DATE_FORMAT.format(date);
};

// A read-only conversation someone shared. It says how old it is, since
// answers can go stale, and invites the reader to ask their own question.
export const SharedChatScreen = ({
  messages,
  title,
  updatedAt,
}: {
  readonly messages: MyUIMessage[];
  readonly title: string;
  readonly updatedAt?: string;
}) => {
  const updated = formatUpdated(updatedAt);

  return (
    <div className="flex h-dvh w-full flex-col bg-background">
      <header className="shrink-0 border-b border-border/60 bg-background pt-[env(safe-area-inset-top)]">
        <div className="mx-auto flex h-14 w-full max-w-3xl items-center gap-3 px-4">
          <a
            aria-label={t('header.title')}
            className="shrink-0 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            href="/"
          >
            <img
              alt=""
              className="h-9 w-9 object-contain"
              height={36}
              src="/logo.png"
              width={36}
            />
          </a>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-bold leading-tight tracking-tight sm:text-lg">
              {title}
            </h1>
            {updated === null ? null : (
              <p
                className="truncate text-xs text-muted-foreground"
                data-testid="shared-updated"
              >
                {t('shared.updated')}: {updated}
              </p>
            )}
          </div>
        </div>
      </header>
      <main
        className="flex min-h-0 flex-1 flex-col [&_p]:text-pretty"
        id="main-content"
        tabIndex={-1}
      >
        {messages.length === 0 ? (
          <p className="m-auto px-4 text-center text-sm text-muted-foreground">
            {t('shared.empty')}
          </p>
        ) : (
          <Thread
            messages={messages}
            status="ready"
          />
        )}
      </main>
      <footer className="shrink-0 border-t border-border/60 bg-background pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3">
        <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4">
          <p className="text-sm text-muted-foreground">{t('shared.ctaHint')}</p>
          <Button asChild>
            <a
              data-testid="shared-cta"
              href="/"
            >
              {t('shared.cta')}
              <ArrowRight aria-hidden="true" />
            </a>
          </Button>
        </div>
      </footer>
    </div>
  );
};

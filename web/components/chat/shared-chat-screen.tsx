'use client';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { t } from '@/lib/i18n';

export const SharedChatScreen = ({
  messages,
  title,
}: {
  readonly messages: MyUIMessage[];
  readonly title: string;
}) => (
  <div className="flex h-dvh w-full flex-col bg-background">
    <header className="shrink-0 border-b border-border/60 bg-background pt-[env(safe-area-inset-top)]">
      <div className="mx-auto flex h-14 w-full max-w-3xl items-center gap-3 px-4">
        <img
          alt="ФИНКИ Хаб"
          className="h-9 w-9 shrink-0 object-contain"
          height={36}
          src="/logo.png"
          width={36}
        />
        <h1 className="min-w-0 flex-1 truncate text-base font-bold leading-tight tracking-tight sm:text-lg">
          {title}
        </h1>
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
  </div>
);

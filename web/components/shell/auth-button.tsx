'use client';

import { LogIn, LogOut } from 'lucide-react';
import { signIn, signOut, useSession } from 'next-auth/react';

import { IconButton } from '@/components/ui/icon-controls';
import { t } from '@/lib/i18n';

export const AuthButton = () => {
  const { data: session, status } = useSession();

  if (status === 'authenticated') {
    const userLabel =
      session.user?.name ?? session.user?.email ?? t('auth.signOut');

    return (
      <IconButton
        aria-label={`${t('auth.signOut')}: ${userLabel}`}
        className="w-auto gap-2 px-3"
        onClick={() => {
          void signOut();
        }}
        title={session.user?.email ?? t('auth.signOut')}
      >
        <LogOut
          aria-hidden="true"
          className="h-5 w-5"
        />
        <span className="block min-w-0 max-w-16 truncate sm:max-w-32">
          {userLabel}
        </span>
      </IconButton>
    );
  }

  return (
    <IconButton
      aria-label={t('auth.signIn')}
      className="w-auto gap-2 px-3"
      disabled={status === 'loading'}
      onClick={() => {
        void signIn();
      }}
      title={t('auth.signIn')}
    >
      <LogIn
        aria-hidden="true"
        className="h-5 w-5"
      />
      <span className="hidden sm:inline">{t('auth.signIn')}</span>
    </IconButton>
  );
};

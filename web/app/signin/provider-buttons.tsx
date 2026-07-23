'use client';

import type { IconType } from 'react-icons';

import { ArrowRight } from 'lucide-react';
import { signIn } from 'next-auth/react';
import { useRef, useState } from 'react';
import { BsGoogle, BsMicrosoft } from 'react-icons/bs';

import type { AuthProviderId } from '@/auth';

import { fireAndForget } from '@/lib/async';

type AuthProvider = {
  readonly id: AuthProviderId;
  readonly name: string;
};

type ProviderButtonsProps = {
  readonly callbackUrl: string;
  readonly providers: readonly AuthProvider[];
};

const providerIcons = {
  google: BsGoogle,
  'microsoft-entra-id': BsMicrosoft,
} satisfies Record<AuthProviderId, IconType>;

export const ProviderButtons = ({
  callbackUrl,
  providers,
}: ProviderButtonsProps) => {
  const signInPendingRef = useRef(false);
  const [isPending, setIsPending] = useState(false);

  const clearPending = (): void => {
    signInPendingRef.current = false;
    setIsPending(false);
  };

  const beginSignIn = async (providerId: AuthProviderId): Promise<void> => {
    if (signInPendingRef.current) {
      return;
    }
    signInPendingRef.current = true;
    setIsPending(true);
    try {
      await signIn(providerId, {
        redirectTo: callbackUrl,
      });
    } catch (error: unknown) {
      clearPending();
      throw error;
    }
  };

  return providers.map((provider) => {
    const ProviderIcon = providerIcons[provider.id];

    return (
      <button
        aria-busy={isPending}
        className="group flex w-full items-center justify-between rounded-2xl border border-border bg-card px-4 py-3 text-left text-sm font-medium transition-[border-color,background-color,transform] hover:border-primary/60 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:translate-y-px disabled:cursor-wait disabled:opacity-70 motion-reduce:transform-none motion-reduce:transition-none"
        disabled={isPending}
        key={provider.id}
        onClick={() => {
          fireAndForget(beginSignIn(provider.id));
        }}
        type="button"
      >
        <span className="flex items-center gap-3">
          <ProviderIcon
            aria-hidden="true"
            className="h-4 w-4 shrink-0"
          />
          <span>Продолжи со {provider.name}</span>
        </span>
        <ArrowRight
          aria-hidden="true"
          className="h-4 w-4 text-muted-foreground group-hover:text-primary motion-safe:transition-[color,transform] motion-safe:group-hover:translate-x-0.5"
        />
      </button>
    );
  });
};

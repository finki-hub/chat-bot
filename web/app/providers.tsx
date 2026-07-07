'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SessionProvider } from 'next-auth/react';
import { posthog } from 'posthog-js';
import { PostHogProvider } from 'posthog-js/react';
import { type ReactNode, useEffect, useState } from 'react';

import { TooltipProvider } from '@/components/ui/tooltip';
import { fireAndForget } from '@/lib/async';
import { useUiStore } from '@/lib/ui-store';

export type ProvidersProps = {
  children: ReactNode;
};

export const Providers = ({ children }: ProvidersProps) => {
  const [queryClient] = useState(() => new QueryClient());
  const [hydrated, setHydrated] = useState(() =>
    useUiStore.persist.hasHydrated(),
  );

  useEffect(() => {
    let mounted = true;

    const rehydrate = async (): Promise<void> => {
      await useUiStore.persist.rehydrate();
      if (mounted) {
        setHydrated(true);
      }
    };

    fireAndForget(rehydrate());

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <SessionProvider>
      <PostHogProvider client={posthog}>
        <QueryClientProvider client={queryClient}>
          <TooltipProvider>{hydrated ? children : null}</TooltipProvider>
        </QueryClientProvider>
      </PostHogProvider>
    </SessionProvider>
  );
};

export default Providers;

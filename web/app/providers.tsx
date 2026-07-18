'use client';

import {
  QueryClient,
  QueryClientProvider,
  useQueryClient,
} from '@tanstack/react-query';
import { SessionProvider, useSession } from 'next-auth/react';
import { posthog } from 'posthog-js';
import { PostHogProvider } from 'posthog-js/react';
import { type ReactNode, useEffect, useRef, useState } from 'react';

import { TooltipProvider } from '@/components/ui/tooltip';
import { fireAndForget } from '@/lib/async';
import { useUiStore } from '@/lib/ui-store';
import { CREDENTIALS_QUERY_KEY } from '@/lib/use-credentials';
import { getModelsSessionKey, MODELS_QUERY_KEY } from '@/lib/use-models';

export type ProvidersProps = {
  children: ReactNode;
};

const SessionScopedModelCache = () => {
  const queryClient = useQueryClient();
  const { data: session, status } = useSession();
  const sessionKey = getModelsSessionKey(status, session);
  const previousSessionKeyRef = useRef<null | string>(null);

  useEffect(() => {
    if (status === 'loading') {
      return;
    }
    const previous = previousSessionKeyRef.current;
    if (previous !== null && previous !== sessionKey) {
      queryClient.removeQueries({
        predicate: (query) =>
          (query.queryKey[0] === MODELS_QUERY_KEY[0] ||
            query.queryKey[0] === CREDENTIALS_QUERY_KEY[0]) &&
          query.queryKey[1] !== sessionKey,
      });
      useUiStore.getState().resetModel();
    }
    previousSessionKeyRef.current = sessionKey;
  }, [queryClient, sessionKey, status]);

  return null;
};

export const Providers = ({ children }: ProvidersProps) => {
  const [queryClient] = useState(() => new QueryClient());
  const [hydrated, setHydrated] = useState(() =>
    useUiStore.persist.hasHydrated(),
  );

  useEffect(() => {
    let mounted = true;

    const rehydrate = async (): Promise<void> => {
      try {
        await useUiStore.persist.rehydrate();
      } finally {
        if (mounted) {
          setHydrated(true);
        }
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
          <SessionScopedModelCache />
          <TooltipProvider>{hydrated ? children : null}</TooltipProvider>
        </QueryClientProvider>
      </PostHogProvider>
    </SessionProvider>
  );
};

export default Providers;

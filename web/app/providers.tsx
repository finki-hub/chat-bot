'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { posthog } from 'posthog-js';
import { PostHogProvider } from 'posthog-js/react';
import { type ReactNode, useEffect, useState } from 'react';

import { TooltipProvider } from '@/components/ui/tooltip';
import { useUiStore } from '@/lib/ui-store';

export type ProvidersProps = {
  children: ReactNode;
};

export const Providers = ({ children }: ProvidersProps) => {
  const [queryClient] = useState(() => new QueryClient());

  useEffect(() => {
    void useUiStore.persist.rehydrate();
  }, []);

  return (
    <PostHogProvider client={posthog}>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>{children}</TooltipProvider>
      </QueryClientProvider>
    </PostHogProvider>
  );
};

export default Providers;

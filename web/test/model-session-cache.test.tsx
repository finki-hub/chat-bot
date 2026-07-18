import type { Session } from 'next-auth';
import type { ReactNode } from 'react';

import {
  QueryClient,
  QueryClientProvider,
  useQueryClient,
} from '@tanstack/react-query';
import { act, render, screen, waitFor } from '@testing-library/react';
import { type SessionContextValue, useSession } from 'next-auth/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Providers } from '@/app/providers';
import { DEFAULT_MODEL, useUiStore } from '@/lib/ui-store';
import { useModels } from '@/lib/use-models';

vi.mock('next-auth/react', () => ({
  SessionProvider: ({ children }: { readonly children: ReactNode }) => (
    <>{children}</>
  ),
  useSession: vi.fn<() => SessionContextValue>(),
}));

vi.mock('posthog-js', () => ({ posthog: {} }));
vi.mock('posthog-js/react', () => ({
  PostHogProvider: ({ children }: { readonly children: ReactNode }) => (
    <>{children}</>
  ),
}));

const useSessionMock = vi.mocked(useSession);

const updateSession = vi.fn<() => Promise<null | Session>>();

const SESSION_A: Session = {
  expires: '2099-01-01T00:00:00.000Z',
  user: { provider: 'google', providerSubject: 'subject-a' },
};
const SESSION_B: Session = {
  expires: '2099-01-01T00:00:00.000Z',
  user: { provider: 'google', providerSubject: 'subject-b' },
};

const catalog = (model: string): Response =>
  Response.json({
    models: [
      {
        availability: 'sponsored',
        id: model,
        name: model,
        provider: 'openai',
      },
    ],
    source: 'live',
    version: 1,
  });

const ModelsProbe = () => {
  const { models } = useModels();
  return <output data-testid="models">{models[0]?.id ?? 'empty'}</output>;
};

describe('session-scoped model cache', () => {
  beforeEach(() => {
    useSessionMock.mockReturnValue({
      data: SESSION_A,
      status: 'authenticated',
      update: updateSession,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('fetches a distinct catalog when the authenticated session changes', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValueOnce(catalog('model-a'))
        .mockResolvedValueOnce(catalog('model-b')),
    );
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const view = render(
      <QueryClientProvider client={queryClient}>
        <ModelsProbe />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('models')).toHaveTextContent('model-a');
    });

    useSessionMock.mockReturnValue({
      data: SESSION_B,
      status: 'authenticated',
      update: updateSession,
    });
    view.rerender(
      <QueryClientProvider client={queryClient}>
        <ModelsProbe />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('models')).toHaveTextContent('model-b');
    });

    expect(fetch).toHaveBeenCalledTimes(2);
    expect(
      queryClient.getQueryData(['models', 'google:subject-a']),
    ).toStrictEqual(
      expect.objectContaining({
        models: [expect.objectContaining({ id: 'model-a' })],
      }),
    );
    expect(
      queryClient.getQueryData(['models', 'google:subject-b']),
    ).toStrictEqual(
      expect.objectContaining({
        models: [expect.objectContaining({ id: 'model-b' })],
      }),
    );
  });

  it('removes the prior catalog and resets selection when the session changes', async () => {
    let queryClient: QueryClient | undefined;
    const CacheProbe = () => {
      const client = useQueryClient();
      queryClient = client;
      return null;
    };
    useUiStore.setState({ model: 'model-from-session-a' });
    useSessionMock.mockReturnValue({
      data: SESSION_A,
      status: 'authenticated',
      update: updateSession,
    });
    const view = render(
      <Providers>
        <CacheProbe />
      </Providers>,
    );

    await waitFor(() => {
      expect(queryClient).toBeDefined();
    });

    act(() => {
      queryClient?.setQueryData(['models', 'google:subject-a'], {
        models: [{ id: 'model-from-session-a' }],
      });
    });

    useSessionMock.mockReturnValue({
      data: SESSION_B,
      status: 'authenticated',
      update: updateSession,
    });
    view.rerender(
      <Providers>
        <CacheProbe />
      </Providers>,
    );

    await waitFor(() => {
      expect(
        queryClient?.getQueryData(['models', 'google:subject-a']),
      ).toBeUndefined();
    });

    expect(useUiStore.getState().model).toBe(DEFAULT_MODEL);
  });
});

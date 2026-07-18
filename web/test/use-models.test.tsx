import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useModels } from '@/lib/use-models';

type AnonymousSession = {
  readonly data: null;
  readonly status: 'unauthenticated';
  readonly update: () => Promise<null>;
};

const authMocks = vi.hoisted(() => ({
  useSession: vi.fn<() => AnonymousSession>(),
}));

vi.mock('next-auth/react', () => authMocks);

const ModelsProbe = () => {
  const { isError, models } = useModels();
  return (
    <div data-testid="models-state">
      {isError ? 'error' : 'ready'}:{models.length}
    </div>
  );
};

const noSession = {
  data: null,
  status: 'unauthenticated',
  update: () => Promise.resolve(null),
} satisfies AnonymousSession;

describe('useModels', () => {
  beforeEach(() => {
    authMocks.useSession.mockReturnValue({
      ...noSession,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reports a typed error catalog as an error state', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          Response.json({ models: [], source: 'error', version: 1 }),
        ),
    );
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ModelsProbe />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('models-state')).toHaveTextContent('error:0');
    });
  });
});

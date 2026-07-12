import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useModels } from '@/lib/use-models';

const ModelsProbe = () => {
  const { isError, models } = useModels();
  return (
    <div data-testid="models-state">
      {isError ? 'error' : 'ready'}:{models.length}
    </div>
  );
};

describe('useModels', () => {
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

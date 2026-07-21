import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor } from '@testing-library/react';
import { useEffect } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useHealth } from '@/lib/use-health';

type HealthProbeProps = {
  readonly onRendered?: (queryClient: QueryClient) => void;
  readonly queryClient: QueryClient;
};

const HealthProbe = ({ onRendered, queryClient }: HealthProbeProps) => {
  const { unavailable } = useHealth();

  useEffect(() => {
    onRendered?.(queryClient);
  });

  return <div data-testid="health-state">{unavailable ? 'down' : 'up'}</div>;
};

const renderHealthProbe = (onRendered?: (queryClient: QueryClient) => void) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retryDelay: 0 } },
  });

  const view = render(
    <QueryClientProvider client={queryClient}>
      <HealthProbe
        onRendered={onRendered}
        queryClient={queryClient}
      />
    </QueryClientProvider>,
  );

  return { queryClient, ...view };
};

const healthResponse = (status: number): Response =>
  Response.json({ ok: status === 200 }, { status });

describe('useHealth', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('keeps the outage banner visible until two recovery checks succeed', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(healthResponse(503))
      .mockResolvedValueOnce(healthResponse(503))
      .mockResolvedValueOnce(healthResponse(200))
      .mockResolvedValueOnce(healthResponse(200));
    vi.stubGlobal('fetch', fetchMock);

    let resolveFirstRecovery: (() => void) | undefined;
    const { queryClient } = renderHealthProbe((client) => {
      const state = client.getQueryState(['health']);
      if (state?.fetchStatus === 'idle' && state.status === 'success') {
        resolveFirstRecovery?.();
        resolveFirstRecovery = undefined;
      }
    });

    await waitFor(() => {
      expect(screen.getByTestId('health-state')).toHaveTextContent('down');
    });

    const firstRecovery = new Promise<void>((resolve) => {
      resolveFirstRecovery = resolve;
    });

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ['health'] });
      await firstRecovery;
    });

    expect(screen.getByTestId('health-state')).toHaveTextContent('down');

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ['health'] });
    });

    expect(fetchMock).toHaveBeenCalledTimes(4);

    await waitFor(() => {
      expect(screen.getByTestId('health-state')).toHaveTextContent('up');
    });
  });
});

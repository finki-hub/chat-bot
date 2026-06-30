import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useHealth } from '@/lib/use-health';

const HealthProbe = () => {
  const { unavailable } = useHealth();

  return <div data-testid="health-state">{unavailable ? 'down' : 'up'}</div>;
};

const renderHealthProbe = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retryDelay: 0 } },
  });

  const view = render(
    <QueryClientProvider client={queryClient}>
      <HealthProbe />
    </QueryClientProvider>,
  );

  return { queryClient, ...view };
};

const healthResponse = (status: number): Response =>
  Response.json({ ok: status === 200 }, { status });

const nextCheckTick = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 1);
  });

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

    const { queryClient } = renderHealthProbe();

    await waitFor(() => {
      expect(screen.getByTestId('health-state')).toHaveTextContent('down');
    });

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ['health'] });
    });

    expect(screen.getByTestId('health-state')).toHaveTextContent('down');

    await act(async () => {
      await nextCheckTick();
    });

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ['health'] });
    });

    expect(fetchMock).toHaveBeenCalledTimes(4);

    await waitFor(() => {
      expect(screen.getByTestId('health-state')).toHaveTextContent('up');
    });
  });
});

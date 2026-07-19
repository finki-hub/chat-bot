import type { ReactNode } from 'react';

import { act, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const renderProviders = async () => {
  const { Providers } = await import('@/app/providers');

  await act(async () => {
    render(
      <Providers>
        <div>ready</div>
      </Providers>,
    );
    await Promise.resolve();
  });
};

describe('Providers', () => {
  afterEach(() => {
    vi.resetModules();
    vi.unstubAllGlobals();
  });

  it('renders children when UI store rehydration fails', async () => {
    const rehydrate = vi
      .fn<() => Promise<void>>()
      .mockRejectedValue(new Error('corrupt storage'));

    vi.stubGlobal('reportError', vi.fn<(error: unknown) => void>());
    vi.doMock('next-auth/react', () => ({
      SessionProvider: ({ children }: { readonly children: ReactNode }) => (
        <>{children}</>
      ),
      useSession: vi.fn<
        () => { readonly data: null; readonly status: 'unauthenticated' }
      >(() => ({
        data: null,
        status: 'unauthenticated',
      })),
    }));
    vi.doMock('posthog-js/react', () => ({
      PostHogProvider: ({ children }: { readonly children: ReactNode }) => (
        <>{children}</>
      ),
    }));

    vi.doMock('@/lib/ui-store', () => ({
      useUiStore: {
        persist: {
          hasHydrated: () => false,
          rehydrate,
        },
      },
    }));

    await renderProviders();

    await waitFor(() => {
      expect(screen.getByText('ready')).toBeInTheDocument();
    });

    expect(rehydrate).toHaveBeenCalledOnce();
  });
});

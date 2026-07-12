import type { ReactNode } from 'react';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ChatCredentialPublic } from '@/lib/api-types';

import { useCredentials } from '@/lib/use-credentials';

const USER_ID = '00000000-0000-4000-8000-000000000001';
const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';
const OPENAI_CREDENTIAL: ChatCredentialPublic = {
  [BASE_URL_FIELD]: null,
  [HAS_API_KEY_FIELD]: true,
  provider: 'openai',
  [USER_ID_FIELD]: USER_ID,
};

const { loadCredentialsMock } = vi.hoisted(() => ({
  loadCredentialsMock:
    vi.fn<
      (signal: AbortSignal) => Promise<null | readonly ChatCredentialPublic[]>
    >(),
}));

vi.mock('@/components/shell/credential-settings-client', () => ({
  loadCredentials: loadCredentialsMock,
}));

const makeWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { readonly children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useCredentials', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('exposes saved credentials when loading succeeds', async () => {
    loadCredentialsMock.mockResolvedValue([OPENAI_CREDENTIAL]);

    const { result } = renderHook(() => useCredentials(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.credentials).toStrictEqual([OPENAI_CREDENTIAL]);
    expect(result.current.isError).toBe(false);
  });

  it('fails closed when credentials cannot be loaded', async () => {
    loadCredentialsMock.mockResolvedValue(null);

    const { result } = renderHook(() => useCredentials(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.credentials).toStrictEqual([]);
    expect(result.current.isError).toBe(true);
  });
});

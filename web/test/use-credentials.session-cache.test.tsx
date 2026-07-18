import type { ReactNode } from 'react';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ChatCredentialPublic } from '@/lib/api-types';

import { useCredentials } from '@/lib/use-credentials';

const USER_A = 'user-a';
const USER_B = 'user-b';
const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';
const credential = (userId: string): ChatCredentialPublic => ({
  [BASE_URL_FIELD]: null,
  [HAS_API_KEY_FIELD]: true,
  provider: 'openai',
  [USER_ID_FIELD]: userId,
});

const { loadCredentialsMock, sessionMock } = vi.hoisted(() => ({
  loadCredentialsMock:
    vi.fn<(signal: AbortSignal) => Promise<ChatCredentialPublic[]>>(),
  sessionMock: vi.fn<
    () => {
      readonly data: {
        readonly user: {
          readonly provider: string;
          readonly providerSubject: string;
        };
      };
      readonly status: 'authenticated';
    }
  >(),
}));

vi.mock('@/components/shell/credential-settings-client', () => ({
  loadCredentials: loadCredentialsMock,
}));

vi.mock('next-auth/react', () => ({
  useSession: sessionMock,
}));

const wrapperFor =
  (queryClient: QueryClient) =>
  ({ children }: { readonly children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

describe('useCredentials session cache', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionMock.mockReturnValue({
      data: { user: { provider: 'test', providerSubject: USER_A } },
      status: 'authenticated',
    });
    loadCredentialsMock.mockResolvedValue([credential(USER_A)]);
  });

  it('does not reuse one authenticated user credentials for another user', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { rerender, result } = renderHook(() => useCredentials(), {
      wrapper: wrapperFor(queryClient),
    });

    await waitFor(() => {
      expect(result.current.credentials).toStrictEqual([credential(USER_A)]);
    });

    sessionMock.mockReturnValue({
      data: { user: { provider: 'test', providerSubject: USER_B } },
      status: 'authenticated',
    });
    loadCredentialsMock.mockResolvedValueOnce([credential(USER_B)]);
    rerender();

    await waitFor(() => {
      expect(result.current.credentials).toStrictEqual([credential(USER_B)]);
    });

    expect(loadCredentialsMock).toHaveBeenCalledTimes(2);
  });
});

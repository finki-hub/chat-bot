import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { authConfiguredMock, authMock, upsertChatUserMock } = vi.hoisted(() => ({
  authConfiguredMock: vi.fn<() => boolean>().mockReturnValue(true),
  authMock: vi.fn<() => Promise<unknown>>(),
  upsertChatUserMock:
    vi.fn<(input: unknown) => Promise<{ readonly id: string }>>(),
}));

vi.mock('@/auth', () => ({
  auth: authMock,
  isAuthConfigured: authConfiguredMock,
}));
vi.mock('@/lib/chat-state-client', () => ({
  createChatStateClient: () => ({ upsertChatUser: upsertChatUserMock }),
}));

describe('getAuthenticatedChatUserId', () => {
  beforeEach(() => {
    vi.resetModules();
    authConfiguredMock.mockReset();
    authConfiguredMock.mockReturnValue(true);
    authMock.mockReset();
    upsertChatUserMock.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('upserts the Auth.js provider identity and returns the API user id', async () => {
    authMock.mockResolvedValue({
      user: {
        email: 'student@example.com',
        image: 'https://example.com/a.png',
        name: 'Student',
        provider: 'microsoft-entra-id',
        providerSubject: 'microsoft-sub-1',
      },
    });
    upsertChatUserMock.mockResolvedValue({
      id: '00000000-0000-4000-8000-000000000001',
    });

    const { getAuthenticatedChatUserId } =
      await import('@/lib/authenticated-chat-user');

    await expect(getAuthenticatedChatUserId()).resolves.toBe(
      '00000000-0000-4000-8000-000000000001',
    );
    expect(upsertChatUserMock).toHaveBeenCalledWith({
      avatarUrl: 'https://example.com/a.png',
      email: 'student@example.com',
      name: 'Student',
      provider: 'microsoft-entra-id',
      providerSubject: 'microsoft-sub-1',
    });
  });

  it('rejects a missing session before touching the API', async () => {
    authMock.mockResolvedValue(null);

    const { AuthenticationRequiredError, getAuthenticatedChatUserId } =
      await import('@/lib/authenticated-chat-user');

    await expect(getAuthenticatedChatUserId()).rejects.toBeInstanceOf(
      AuthenticationRequiredError,
    );
    expect(upsertChatUserMock).not.toHaveBeenCalled();
  });
});

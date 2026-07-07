import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { authMock, upsertGoogleUserMock } = vi.hoisted(() => ({
  authMock: vi.fn<() => Promise<unknown>>(),
  upsertGoogleUserMock:
    vi.fn<(input: unknown) => Promise<{ readonly id: string }>>(),
}));

vi.mock('@/auth', () => ({ auth: authMock }));
vi.mock('@/lib/chat-state-client', () => ({
  createChatStateClient: () => ({ upsertGoogleUser: upsertGoogleUserMock }),
}));

describe('getAuthenticatedChatUserId', () => {
  beforeEach(() => {
    vi.resetModules();
    authMock.mockReset();
    upsertGoogleUserMock.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('upserts the Auth.js Google subject and returns the API user id', async () => {
    authMock.mockResolvedValue({
      user: {
        email: 'student@example.com',
        googleSubject: 'google-sub-1',
        image: 'https://example.com/a.png',
        name: 'Student',
      },
    });
    upsertGoogleUserMock.mockResolvedValue({
      id: '00000000-0000-4000-8000-000000000001',
    });

    const { getAuthenticatedChatUserId } =
      await import('@/lib/authenticated-chat-user');

    await expect(getAuthenticatedChatUserId()).resolves.toBe(
      '00000000-0000-4000-8000-000000000001',
    );
    expect(upsertGoogleUserMock).toHaveBeenCalledWith({
      avatarUrl: 'https://example.com/a.png',
      email: 'student@example.com',
      name: 'Student',
      providerSubject: 'google-sub-1',
    });
  });

  it('rejects a missing session before touching the API', async () => {
    authMock.mockResolvedValue(null);

    const { AuthenticationRequiredError, getAuthenticatedChatUserId } =
      await import('@/lib/authenticated-chat-user');

    await expect(getAuthenticatedChatUserId()).rejects.toBeInstanceOf(
      AuthenticationRequiredError,
    );
    expect(upsertGoogleUserMock).not.toHaveBeenCalled();
  });
});

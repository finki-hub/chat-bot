import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

describe('createChatStateClient', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('upserts a Google user with the server API key', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json(
        Object.fromEntries([
          ['avatar_url', 'https://example.com/a.png'],
          ['email', 'student@example.com'],
          ['id', '00000000-0000-4000-8000-000000000001'],
          ['name', 'Student'],
          ['provider', 'google'],
          ['provider_subject', 'google-sub-1'],
        ]),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { createChatStateClient } = await import('@/lib/chat-state-client');
    const user = await createChatStateClient().upsertGoogleUser({
      avatarUrl: 'https://example.com/a.png',
      email: 'student@example.com',
      name: 'Student',
      providerSubject: 'google-sub-1',
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('https://api:8880/chat/state/users/google');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('x-api-key')).toBe('test-key');
    expect(JSON.parse(init.body as string)).toStrictEqual(
      Object.fromEntries([
        ['avatar_url', 'https://example.com/a.png'],
        ['email', 'student@example.com'],
        ['name', 'Student'],
        ['provider_subject', 'google-sub-1'],
      ]),
    );
    expect(user.id).toBe('00000000-0000-4000-8000-000000000001');
  });
});

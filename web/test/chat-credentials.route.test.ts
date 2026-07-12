import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import { ChatStateRequestError as ChatStateRequestErrorForTest } from '@/lib/chat-state-client';

/* eslint-disable camelcase -- Route tests assert the Python API wire contract. */

type DeleteCredentialInput = {
  readonly provider: ChatCredentialProvider;
  readonly userId: string;
};

type ListCredentialsInput = {
  readonly userId: string;
};

type UpsertCredentialInput = {
  readonly apiKey: string;
  readonly baseUrl?: null | string;
  readonly provider: ChatCredentialProvider;
  readonly userId: string;
};

const { getAuthenticatedChatUserIdMock, stateClientMock } = vi.hoisted(() => ({
  getAuthenticatedChatUserIdMock: vi.fn<() => Promise<string>>(),
  stateClientMock: {
    deleteCredential: vi.fn<(input: DeleteCredentialInput) => Promise<void>>(),
    listCredentials:
      vi.fn<
        (
          input: ListCredentialsInput,
        ) => Promise<readonly ChatCredentialPublic[]>
      >(),
    upsertCredential:
      vi.fn<(input: UpsertCredentialInput) => Promise<ChatCredentialPublic>>(),
  },
}));

vi.mock('@/lib/authenticated-chat-user', () => {
  class AuthenticationRequiredError extends Error {
    constructor() {
      super('Authentication required');
      this.name = 'AuthenticationRequiredError';
    }
  }

  return {
    AuthenticationRequiredError,
    getAuthenticatedChatUserId: getAuthenticatedChatUserIdMock,
  };
});

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'https://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'https://api:8880', CHAT_API_KEY: 'test-key' },
}));

vi.mock('@/lib/chat-state-client', () => {
  class ChatStateRequestError extends Error {
    readonly status: number;

    constructor(status: number, options?: ErrorOptions) {
      super('Chat state request failed', options);
      this.name = 'ChatStateRequestError';
      this.status = status;
    }
  }

  return {
    ChatStateRequestError,
    createChatStateClient: () => stateClientMock,
  };
});

const USER_ID = '00000000-0000-4000-8000-000000000001';

const jsonRequest = (body: unknown): Request =>
  new Request('https://localhost/api/chat/credentials', {
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
    method: 'PUT',
  });

describe('/api/chat/credentials', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    getAuthenticatedChatUserIdMock.mockResolvedValue(USER_ID);
    stateClientMock.listCredentials.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('lists credentials for the authenticated chat user only', async () => {
    const credential: ChatCredentialPublic = {
      base_url: null,
      has_api_key: true,
      provider: 'anthropic',
      user_id: USER_ID,
    };
    stateClientMock.listCredentials.mockResolvedValueOnce([credential]);
    const { GET } = await import('@/app/api/chat/credentials/route');

    const res = await GET();

    await expect(res.json()).resolves.toStrictEqual([credential]);
    expect(stateClientMock.listCredentials).toHaveBeenCalledWith({
      userId: USER_ID,
    });
  });

  it('upserts credentials for the authenticated chat user only', async () => {
    const credential: ChatCredentialPublic = {
      base_url: 'https://api.openai.com/v1',
      has_api_key: true,
      provider: 'openai',
      user_id: USER_ID,
    };
    stateClientMock.upsertCredential.mockResolvedValueOnce(credential);
    const { PUT } = await import('@/app/api/chat/credentials/route');

    const res = await PUT(
      jsonRequest({
        api_key: 'user-secret-key',
        base_url: 'https://api.openai.com/v1',
        provider: 'openai',
      }),
    );

    await expect(res.json()).resolves.toStrictEqual(credential);
    expect(stateClientMock.upsertCredential).toHaveBeenCalledWith({
      apiKey: 'user-secret-key',
      baseUrl: 'https://api.openai.com/v1',
      provider: 'openai',
      userId: USER_ID,
    });
  });

  it('identifies a rejected custom credential base URL', async () => {
    stateClientMock.upsertCredential.mockRejectedValueOnce(
      new ChatStateRequestErrorForTest(422),
    );
    const { PUT } = await import('@/app/api/chat/credentials/route');

    const response = await PUT(
      jsonRequest({
        api_key: 'user-secret-key',
        base_url: 'https://openai-proxy.example/v1',
        provider: 'openai',
      }),
    );

    expect(response.status).toBe(422);
    await expect(response.json()).resolves.toStrictEqual({
      error: 'Credential base URL is not allowed',
    });
  });

  it('upserts an Ollama key and endpoint for the authenticated user', async () => {
    const credential: ChatCredentialPublic = {
      base_url: 'https://ollama.com',
      has_api_key: true,
      provider: 'ollama',
      user_id: USER_ID,
    };
    stateClientMock.upsertCredential.mockResolvedValueOnce(credential);
    const { PUT } = await import('@/app/api/chat/credentials/route');

    const res = await PUT(
      jsonRequest({
        api_key: 'ollama-user-key',
        base_url: 'https://ollama.com',
        provider: 'ollama',
      }),
    );

    expect(res.status).toBe(200);
    expect(stateClientMock.upsertCredential).toHaveBeenCalledWith({
      apiKey: 'ollama-user-key',
      baseUrl: 'https://ollama.com',
      provider: 'ollama',
      userId: USER_ID,
    });
  });

  it('deletes credentials for the authenticated chat user only', async () => {
    stateClientMock.deleteCredential.mockResolvedValueOnce(undefined);
    const { DELETE } =
      await import('@/app/api/chat/credentials/[provider]/route');

    const res = await DELETE(new Request('https://localhost'), {
      params: Promise.resolve({ provider: 'google' }),
    });

    expect(res.status).toBe(204);
    expect(stateClientMock.deleteCredential).toHaveBeenCalledWith({
      provider: 'google',
      userId: USER_ID,
    });
  });
});

/* eslint-enable camelcase -- end Python API wire contract assertions. */

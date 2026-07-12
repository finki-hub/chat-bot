import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  ChatCredentialProvider,
  ChatCredentialPublic,
} from '@/lib/api-types';

import { CredentialSettingsDialog } from '@/components/shell/credential-settings-dialog';
import { CREDENTIALS_QUERY_KEY } from '@/lib/use-credentials';

type SaveCredentialInput = {
  readonly apiKey: string;
  readonly baseUrl: string;
  readonly provider: ChatCredentialProvider;
};

const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID = '00000000-0000-4000-8000-000000000001';
const USER_ID_FIELD = 'user_id';
const OPENAI_BASE_URL = 'https://openai-proxy.example/v1';

const openaiCredential = (baseUrl: string): ChatCredentialPublic => ({
  [BASE_URL_FIELD]: baseUrl,
  [HAS_API_KEY_FIELD]: true,
  provider: 'openai',
  [USER_ID_FIELD]: USER_ID,
});

const ollamaCredential = (baseUrl: string): ChatCredentialPublic => ({
  [BASE_URL_FIELD]: baseUrl,
  [HAS_API_KEY_FIELD]: true,
  provider: 'ollama',
  [USER_ID_FIELD]: USER_ID,
});

const anthropicCredential = (): ChatCredentialPublic => ({
  [BASE_URL_FIELD]: null,
  [HAS_API_KEY_FIELD]: true,
  provider: 'anthropic',
  [USER_ID_FIELD]: USER_ID,
});

const { deleteCredentialMock, loadCredentialsMock, saveCredentialMock } =
  vi.hoisted(() => ({
    deleteCredentialMock:
      vi.fn<(provider: ChatCredentialProvider) => Promise<boolean>>(),
    loadCredentialsMock:
      vi.fn<
        (signal: AbortSignal) => Promise<null | readonly ChatCredentialPublic[]>
      >(),
    saveCredentialMock:
      vi.fn<
        (input: SaveCredentialInput) => Promise<ChatCredentialPublic | null>
      >(),
  }));

vi.mock('@/components/shell/credential-settings-client', () => ({
  deleteCredential: deleteCredentialMock,
  loadCredentials: loadCredentialsMock,
  saveCredential: saveCredentialMock,
}));

const renderDialog = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={queryClient}>
      <CredentialSettingsDialog
        onOpenChange={vi.fn<(open: boolean) => void>()}
        open
      />
    </QueryClientProvider>,
  );
  return { queryClient, ...view };
};

describe('CredentialSettingsDialog', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    deleteCredentialMock.mockResolvedValue(true);
    loadCredentialsMock.mockResolvedValue([openaiCredential(OPENAI_BASE_URL)]);
    saveCredentialMock.mockResolvedValue(openaiCredential(OPENAI_BASE_URL));
  });

  it('preserves a saved custom base URL when only the API key changes', async () => {
    const { queryClient } = renderDialog();

    const keyInput = await screen.findByLabelText('OpenAI API key');
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error('OpenAI credential form not found');
    }

    fireEvent.change(keyInput, { target: { value: 'replacement-key' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Зачувај' }));

    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledWith({
        apiKey: 'replacement-key',
        baseUrl: OPENAI_BASE_URL,
        provider: 'openai',
      });
    });

    expect(queryClient.getQueryData(CREDENTIALS_QUERY_KEY)).toStrictEqual([
      openaiCredential(OPENAI_BASE_URL),
    ]);
    expect(loadCredentialsMock).toHaveBeenCalledTimes(2);
  });

  it('clears a saved custom base URL when the credential is deleted', async () => {
    loadCredentialsMock
      .mockResolvedValueOnce([openaiCredential(OPENAI_BASE_URL)])
      .mockResolvedValueOnce([]);
    const { queryClient } = renderDialog();

    const baseUrlInput = await screen.findByLabelText('OpenAI base URL');

    expect(baseUrlInput).toHaveValue(OPENAI_BASE_URL);

    fireEvent.click(screen.getByRole('button', { name: 'Избриши' }));

    await waitFor(() => {
      expect(baseUrlInput).toHaveValue('');
    });

    expect(queryClient.getQueryData(CREDENTIALS_QUERY_KEY)).toStrictEqual([]);
    expect(loadCredentialsMock).toHaveBeenCalledTimes(2);
  });

  it('replaces stale cached providers with the authoritative list after save', async () => {
    loadCredentialsMock
      .mockResolvedValueOnce([
        openaiCredential(OPENAI_BASE_URL),
        anthropicCredential(),
      ])
      .mockResolvedValueOnce([openaiCredential(OPENAI_BASE_URL)]);
    const { queryClient } = renderDialog();
    const keyInput = await screen.findByLabelText('OpenAI API key');
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error('OpenAI credential form not found');
    }

    fireEvent.change(keyInput, { target: { value: 'replacement-key' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Зачувај' }));

    await waitFor(() => {
      expect(loadCredentialsMock).toHaveBeenCalledTimes(2);
      expect(queryClient.getQueryData(CREDENTIALS_QUERY_KEY)).toStrictEqual([
        openaiCredential(OPENAI_BASE_URL),
      ]);
    });
  });

  it('prevents editing until a failed credential load is retried', async () => {
    loadCredentialsMock.mockReset();
    loadCredentialsMock
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce([openaiCredential(OPENAI_BASE_URL)]);
    renderDialog();

    await expect(screen.findByRole('alert')).resolves.toHaveTextContent(
      'Клучевите не можеа да се вчитаат.',
    );
    expect(screen.queryByLabelText('OpenAI API key')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Обиди се повторно' }));

    await expect(
      screen.findByLabelText('OpenAI API key'),
    ).resolves.toBeInTheDocument();
  });

  it('saves an Ollama key and custom endpoint', async () => {
    saveCredentialMock.mockResolvedValueOnce(
      ollamaCredential('https://ollama.example'),
    );
    renderDialog();

    const keyInput = await screen.findByLabelText('Ollama API key');
    const baseUrlInput = screen.getByLabelText('Ollama base URL');
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error('Ollama credential form not found');
    }

    fireEvent.change(keyInput, { target: { value: 'ollama-user-key' } });
    fireEvent.change(baseUrlInput, {
      target: { value: 'https://ollama.example' },
    });
    fireEvent.click(within(form).getByRole('button', { name: 'Зачувај' }));

    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledWith({
        apiKey: 'ollama-user-key',
        baseUrl: 'https://ollama.example',
        provider: 'ollama',
      });
    });
  });
});

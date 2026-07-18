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

import { CredentialBaseUrlRejectedError } from '@/components/shell/credential-settings-client';
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
const OPENAI_API_KEY_LABEL = 'OpenAI API key';
const OPENAI_FORM_NOT_FOUND = 'OpenAI credential form not found';
const REPLACEMENT_KEY = 'replacement-key';

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
  CredentialBaseUrlRejectedError: class extends Error {},
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

  it('uses a localized accessible name for the close action', async () => {
    renderDialog();

    await expect(
      screen.findByRole('button', { name: 'Затвори' }),
    ).resolves.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Close' }),
    ).not.toBeInTheDocument();
  });

  it('preserves a saved custom base URL when only the API key changes', async () => {
    const { queryClient } = renderDialog();

    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error(OPENAI_FORM_NOT_FOUND);
    }

    fireEvent.change(keyInput, { target: { value: REPLACEMENT_KEY } });
    fireEvent.click(within(form).getByRole('button', { name: 'Зачувај' }));

    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledWith({
        apiKey: REPLACEMENT_KEY,
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
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error(OPENAI_FORM_NOT_FOUND);
    }

    fireEvent.change(keyInput, { target: { value: REPLACEMENT_KEY } });
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
    expect(
      screen.queryByLabelText(OPENAI_API_KEY_LABEL),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Обиди се повторно' }));

    await expect(
      screen.findByLabelText(OPENAI_API_KEY_LABEL),
    ).resolves.toBeInTheDocument();
  });

  it('reports a save failure without claiming credential loading failed', async () => {
    saveCredentialMock.mockResolvedValueOnce(null);
    renderDialog();

    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error(OPENAI_FORM_NOT_FOUND);
    }

    fireEvent.change(keyInput, { target: { value: REPLACEMENT_KEY } });
    fireEvent.click(within(form).getByRole('button', { name: 'Зачувај' }));

    await expect(
      screen.findByText('Клучот не можеше да се зачува.'),
    ).resolves.toBeInTheDocument();
    expect(
      screen.queryByText('Клучевите не можеа да се вчитаат.'),
    ).not.toBeInTheDocument();
  });

  it('explains how to fix a rejected custom base URL', async () => {
    saveCredentialMock.mockRejectedValueOnce(
      new CredentialBaseUrlRejectedError(),
    );
    renderDialog();

    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const form = keyInput.closest('form');
    if (form === null) {
      throw new Error(OPENAI_FORM_NOT_FOUND);
    }

    fireEvent.change(keyInput, { target: { value: REPLACEMENT_KEY } });
    fireEvent.change(screen.getByLabelText('OpenAI base URL'), {
      target: { value: 'https://openai-proxy.example/v1' },
    });
    fireEvent.click(within(form).getByRole('button', { name: 'Зачувај' }));

    await expect(
      screen.findByText(
        'Base URL адресата не е дозволена. Остави го полето празно за стандардниот endpoint или побарај администраторот да ја додаде.',
      ),
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

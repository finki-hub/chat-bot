import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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

const SESSION_A = 'test:user-a';
const SESSION_B = 'test:user-b';
const OPENAI_API_KEY_LABEL = 'OpenAI API клуч';
const SAVE_CREDENTIALS_LABEL = 'Зачувај клучеви';
const SAVING_CREDENTIALS_LABEL = 'Се зачувуваат клучевите…';
const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';

const credential = (
  provider: ChatCredentialProvider,
  userId: string,
  baseUrl: null | string = null,
): ChatCredentialPublic => ({
  [BASE_URL_FIELD]: baseUrl,
  [HAS_API_KEY_FIELD]: true,
  provider,
  [USER_ID_FIELD]: userId,
});

const {
  deleteCredentialMock,
  loadCredentialsMock,
  refetchModelsMock,
  reportErrorMock,
  saveCredentialMock,
  sessionKeyMock,
} = vi.hoisted(() => ({
  deleteCredentialMock:
    vi.fn<(provider: ChatCredentialProvider) => Promise<boolean>>(),
  loadCredentialsMock:
    vi.fn<
      (signal: AbortSignal) => Promise<null | readonly ChatCredentialPublic[]>
    >(),
  refetchModelsMock: vi.fn<() => Promise<unknown>>(),
  reportErrorMock: vi.fn<(error: unknown) => void>(),
  saveCredentialMock:
    vi.fn<
      (input: SaveCredentialInput) => Promise<ChatCredentialPublic | null>
    >(),
  sessionKeyMock: vi.fn<() => null | string>(),
}));

vi.mock('@/components/shell/credential-settings-client', () => ({
  CredentialBaseUrlRejectedError: class extends Error {},
  deleteCredential: deleteCredentialMock,
  loadCredentials: loadCredentialsMock,
  saveCredential: saveCredentialMock,
}));

vi.mock('@/lib/use-models', () => ({
  getModelsSessionKey: sessionKeyMock,
  useModels: () => ({ refetch: refetchModelsMock }),
}));

vi.mock('next-auth/react', () => ({
  useSession: () => ({ data: null, status: 'unauthenticated' }),
}));

const renderDialog = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const dialog = (open = true) => (
    <QueryClientProvider client={queryClient}>
      <CredentialSettingsDialog
        onOpenChangeAction={vi.fn<(open: boolean) => void>()}
        open={open}
      />
    </QueryClientProvider>
  );
  const view = render(dialog());
  return {
    queryClient,
    ...view,
    rerenderDialog: (open = true) => {
      view.rerender(dialog(open));
    },
  };
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal('reportError', reportErrorMock);
  sessionKeyMock.mockReturnValue(SESSION_A);
  deleteCredentialMock.mockResolvedValue(true);
  loadCredentialsMock.mockResolvedValue([]);
  refetchModelsMock.mockResolvedValue({});
  saveCredentialMock.mockResolvedValue(credential('openai', SESSION_A));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('CredentialSettingsDialog regressions', () => {
  it('disables native validation for optional provider URLs', async () => {
    renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);

    expect(keyInput.closest('form')).toHaveAttribute('novalidate');
  });

  it('links every provider to its official API-key dashboard', async () => {
    renderDialog();
    await screen.findByLabelText(OPENAI_API_KEY_LABEL);

    const providerLinks = [
      ['OpenAI', 'https://platform.openai.com/api-keys'],
      ['Google / Gemini', 'https://aistudio.google.com/api-keys'],
      ['Anthropic', 'https://platform.claude.com/settings/keys'],
      ['Ollama', 'https://ollama.com/settings/keys'],
    ] as const;

    for (const [provider, href] of providerLinks) {
      const link = screen.getByRole('link', {
        name: `Добиј API клуч: ${provider}`,
      });

      expect(link).toHaveAttribute('href', href);
      expect(link).toHaveAttribute('rel', 'noreferrer');
      expect(link).toHaveAttribute('target', '_blank');
    }
  });

  it('clears plaintext drafts when the active session changes', async () => {
    const { rerenderDialog } = renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.change(keyInput, { target: { value: 'user-a-secret' } });

    sessionKeyMock.mockReturnValue(SESSION_B);
    rerenderDialog();

    await waitFor(() => {
      expect(screen.getByLabelText(OPENAI_API_KEY_LABEL)).toHaveValue('');
    });
  });

  it('ignores a delete error after the session changes away and back', async () => {
    const pendingDelete = Promise.withResolvers<boolean>();
    deleteCredentialMock.mockReturnValueOnce(pendingDelete.promise);
    loadCredentialsMock.mockResolvedValueOnce([
      credential('openai', SESSION_A),
    ]);
    const { rerenderDialog } = renderDialog();

    await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.click(screen.getByRole('button', { name: 'Избриши' }));
    fireEvent.click(await screen.findByTestId('confirm-action'));

    sessionKeyMock.mockReturnValue(SESSION_B);
    rerenderDialog();
    sessionKeyMock.mockReturnValue(SESSION_A);
    rerenderDialog();
    const requestError = new TypeError('Delete request failed');
    await act(async () => {
      pendingDelete.reject(requestError);

      await expect(pendingDelete.promise).rejects.toBe(requestError);
    });

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('ignores a save error that settles after the dialog closes', async () => {
    const pendingSave = Promise.withResolvers<ChatCredentialPublic | null>();
    saveCredentialMock.mockReturnValueOnce(pendingSave.promise);
    const { rerenderDialog } = renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.change(keyInput, { target: { value: 'invalid-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );

    rerenderDialog(false);
    await waitFor(() => {
      expect(
        screen.queryByRole('dialog', { name: 'Лични API клучеви' }),
      ).not.toBeInTheDocument();
    });
    rerenderDialog();
    await screen.findByLabelText(OPENAI_API_KEY_LABEL);

    await act(async () => {
      pendingSave.resolve(null);
      await pendingSave.promise;
    });

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('prevents overlapping saves after closing and reopening', async () => {
    const pendingSave = Promise.withResolvers<ChatCredentialPublic | null>();
    saveCredentialMock
      .mockReturnValueOnce(pendingSave.promise)
      .mockResolvedValueOnce(credential('openai', SESSION_A));
    const { rerenderDialog } = renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.change(keyInput, { target: { value: 'first-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );
    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledOnce();
    });

    rerenderDialog(false);
    rerenderDialog();
    fireEvent.change(await screen.findByLabelText(OPENAI_API_KEY_LABEL), {
      target: { value: 'second-secret' },
    });
    const saveButton = screen.getByRole('button', {
      name: SAVING_CREDENTIALS_LABEL,
    });

    expect(saveButton).toBeDisabled();

    fireEvent.click(saveButton);

    expect(saveCredentialMock).toHaveBeenCalledOnce();

    await act(async () => {
      pendingSave.resolve(null);
      await pendingSave.promise;
    });
    await waitFor(() => {
      expect(saveButton).toBeEnabled();
    });

    fireEvent.click(saveButton);
    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledTimes(2);
    });
  });

  it('prevents overlapping saves after unmounting and remounting', async () => {
    const pendingSave = Promise.withResolvers<ChatCredentialPublic | null>();
    saveCredentialMock.mockReturnValueOnce(pendingSave.promise);
    const firstDialog = renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.change(keyInput, { target: { value: 'first-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );
    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledOnce();
    });

    firstDialog.unmount();
    renderDialog();
    const remountedKeyInput =
      await screen.findByLabelText(OPENAI_API_KEY_LABEL);

    expect(remountedKeyInput).toBeDisabled();

    await act(async () => {
      pendingSave.resolve(null);
      await pendingSave.promise;
    });
    await waitFor(() => {
      expect(remountedKeyInput).toBeEnabled();
    });
  });

  it('blocks a replacement save while an earlier delete remains pending', async () => {
    const pendingDelete = Promise.withResolvers<boolean>();
    deleteCredentialMock.mockReturnValueOnce(pendingDelete.promise);
    loadCredentialsMock.mockResolvedValue([credential('openai', SESSION_A)]);
    const { rerenderDialog } = renderDialog();

    await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.click(screen.getByRole('button', { name: 'Избриши' }));
    fireEvent.click(await screen.findByTestId('confirm-action'));
    await waitFor(() => {
      expect(deleteCredentialMock).toHaveBeenCalledOnce();
    });

    sessionKeyMock.mockReturnValue(SESSION_B);
    rerenderDialog();
    sessionKeyMock.mockReturnValue(SESSION_A);
    rerenderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);

    expect(keyInput).toBeDisabled();
    expect(saveCredentialMock).not.toHaveBeenCalled();

    await act(async () => {
      pendingDelete.resolve(false);
      await pendingDelete.promise;
    });
    await waitFor(() => {
      expect(keyInput).toBeEnabled();
    });
  });
});

describe('CredentialSettingsDialog save reconciliation', () => {
  it('updates only the active session credential cache after saving', async () => {
    const userBCredentials = [credential('anthropic', SESSION_B)];
    const { queryClient } = renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    queryClient.setQueryData(
      [...CREDENTIALS_QUERY_KEY, SESSION_B],
      userBCredentials,
    );

    fireEvent.change(keyInput, { target: { value: 'user-a-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );

    await waitFor(() => {
      expect(saveCredentialMock).toHaveBeenCalledOnce();
    });

    expect(
      queryClient.getQueryData([...CREDENTIALS_QUERY_KEY, SESSION_B]),
    ).toStrictEqual(userBCredentials);
  });

  it('ignores save reconciliation after the session changes', async () => {
    const pendingModelRefetch = Promise.withResolvers<unknown>();
    const userABaseUrl = 'https://user-a.example/v1';
    saveCredentialMock.mockResolvedValueOnce(
      credential('openai', SESSION_A, userABaseUrl),
    );
    refetchModelsMock.mockReturnValueOnce(pendingModelRefetch.promise);
    const { rerenderDialog } = renderDialog();
    const keyInput = await screen.findByLabelText(OPENAI_API_KEY_LABEL);

    fireEvent.change(keyInput, { target: { value: 'user-a-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );
    await waitFor(() => {
      expect(refetchModelsMock).toHaveBeenCalledOnce();
    });

    sessionKeyMock.mockReturnValue(SESSION_B);
    rerenderDialog();
    await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    await act(async () => {
      pendingModelRefetch.resolve({});
      await pendingModelRefetch.promise;
    });

    expect(screen.getByLabelText('OpenAI Base URL (опционално)')).toHaveValue(
      '',
    );
  });

  it('reconciles successful saves when another provider fails unexpectedly', async () => {
    const unexpectedError = new RangeError('Unexpected provider response');
    const savedOpenAi = credential('openai', SESSION_A);
    loadCredentialsMock
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([savedOpenAi]);
    saveCredentialMock.mockImplementation(({ provider }) =>
      provider === 'openai'
        ? Promise.resolve(savedOpenAi)
        : Promise.reject(unexpectedError),
    );
    const { queryClient } = renderDialog();

    const openaiKey = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const googleKey = screen.getByLabelText('Google / Gemini API клуч');
    fireEvent.change(openaiKey, { target: { value: 'openai-secret' } });
    fireEvent.change(googleKey, { target: { value: 'google-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );

    await expect(screen.findByRole('alert')).resolves.toHaveTextContent(
      'Клучот не можеше да се зачува.',
    );
    expect(openaiKey).toHaveValue('');
    expect(googleKey).toHaveValue('google-secret');
    expect(
      queryClient.getQueryData([...CREDENTIALS_QUERY_KEY, SESSION_A]),
    ).toStrictEqual([savedOpenAi]);

    await waitFor(() => {
      expect(reportErrorMock).toHaveBeenCalledWith(unexpectedError);
    });
  });

  it('attributes simultaneous save failures to each provider field', async () => {
    saveCredentialMock.mockImplementation(({ provider }) => {
      if (provider === 'google') {
        return Promise.reject(new CredentialBaseUrlRejectedError());
      }
      return Promise.resolve(null);
    });
    renderDialog();

    const openaiKey = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const googleKey = screen.getByLabelText('Google / Gemini API клуч');
    const googleBaseUrl = screen.getByLabelText(
      'Google / Gemini Base URL (опционално)',
    );
    fireEvent.change(openaiKey, { target: { value: 'openai-secret' } });
    fireEvent.change(googleKey, { target: { value: 'google-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );

    await waitFor(() => {
      expect(openaiKey).not.toHaveAttribute('aria-invalid');
      expect(googleBaseUrl).toHaveAttribute('aria-invalid', 'true');
    });

    expect(openaiKey).toHaveAccessibleDescription(
      expect.stringContaining('Клучот не можеше да се зачува.'),
    );

    expect(googleBaseUrl).toHaveAccessibleDescription(
      expect.stringContaining('Base URL адресата не е дозволена.'),
    );

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
      ).toBeEnabled();
    });

    fireEvent.change(googleKey, { target: { value: 'updated-google-secret' } });

    expect(googleBaseUrl).toHaveAttribute('aria-invalid', 'true');

    fireEvent.change(googleBaseUrl, {
      target: { value: 'https://allowed.example/v1' },
    });

    await waitFor(() => {
      expect(googleBaseUrl).not.toHaveAttribute('aria-invalid');
    });
  });

  it('clears a provider save failure after deleting its saved credential', async () => {
    loadCredentialsMock.mockResolvedValueOnce([
      credential('openai', SESSION_A),
    ]);
    saveCredentialMock.mockResolvedValueOnce(null);
    renderDialog();

    const openaiKey = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    fireEvent.change(openaiKey, { target: { value: 'replacement-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );
    await screen.findByRole('alert');

    fireEvent.click(screen.getByRole('button', { name: 'Избриши' }));
    fireEvent.click(await screen.findByTestId('confirm-action'));

    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
  });
});

describe('CredentialSettingsDialog accessibility regressions', () => {
  it('wraps reverse tabbing from the initially focused title', async () => {
    const user = userEvent.setup();
    renderDialog();

    const dialog = await screen.findByRole('dialog', {
      name: 'Лични API клучеви',
    });
    const title = screen.getByRole('heading', { name: 'Лични API клучеви' });
    await waitFor(() => {
      expect(title).toHaveFocus();
    });

    expect(title).toHaveAttribute('tabindex', '-1');

    await user.tab({ shift: true });

    expect(title).not.toHaveFocus();
    expect(dialog.querySelector('[data-slot="dialog-close"]')).toHaveFocus();
  });

  it('keeps a corrected failed field visible in a short dialog', async () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn<(query: string) => MediaQueryList>((query) => ({
        addEventListener: vi.fn<MediaQueryList['addEventListener']>(),
        addListener: vi.fn<MediaQueryList['addListener']>(),
        dispatchEvent: vi
          .fn<MediaQueryList['dispatchEvent']>()
          .mockReturnValue(true),
        matches: query === '(max-height: 30rem)',
        media: query,
        onchange: null,
        removeEventListener: vi.fn<MediaQueryList['removeEventListener']>(),
        removeListener: vi.fn<MediaQueryList['removeListener']>(),
      })),
    );
    saveCredentialMock.mockResolvedValue(null);
    renderDialog();

    const openaiKey = await screen.findByLabelText(OPENAI_API_KEY_LABEL);
    const ollamaKey = screen.getByLabelText('Ollama API клуч');
    const providerScroller = openaiKey.closest('.overflow-y-auto');
    if (!(providerScroller instanceof HTMLElement)) {
      throw new TypeError('Credential provider scroll region not found');
    }
    fireEvent.change(openaiKey, { target: { value: 'openai-secret' } });
    fireEvent.change(ollamaKey, { target: { value: 'ollama-secret' } });
    fireEvent.click(
      screen.getByRole('button', { name: SAVE_CREDENTIALS_LABEL }),
    );
    await screen.findByRole('alert');
    providerScroller.scrollTop = 240;
    ollamaKey.focus();

    fireEvent.change(ollamaKey, { target: { value: 'corrected-secret' } });

    expect(ollamaKey).toHaveFocus();
    expect(providerScroller.scrollTop).toBe(240);
  });
});

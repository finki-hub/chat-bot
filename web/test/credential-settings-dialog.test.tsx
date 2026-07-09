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

type SaveCredentialInput = {
  readonly apiKey: string;
  readonly baseUrl: string;
  readonly provider: ChatCredentialProvider;
};

const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID = '00000000-0000-4000-8000-000000000001';
const USER_ID_FIELD = 'user_id';

const openaiCredential = (baseUrl: string): ChatCredentialPublic => ({
  [BASE_URL_FIELD]: baseUrl,
  [HAS_API_KEY_FIELD]: true,
  provider: 'openai',
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

describe('CredentialSettingsDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    loadCredentialsMock.mockResolvedValue([
      openaiCredential('https://openai-proxy.example/v1'),
    ]);
    saveCredentialMock.mockResolvedValue(
      openaiCredential('https://openai-proxy.example/v1'),
    );
  });

  it('preserves a saved custom base URL when only the API key changes', async () => {
    render(
      <CredentialSettingsDialog
        onOpenChange={vi.fn<(open: boolean) => void>()}
        open
      />,
    );

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
        baseUrl: 'https://openai-proxy.example/v1',
        provider: 'openai',
      });
    });
  });
});

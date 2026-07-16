import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const SHARE_TOKEN = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d23';
const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue();

describe('ShareConversationButton', () => {
  beforeEach(() => {
    writeText.mockClear();
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          Response.json({ shareToken: SHARE_TOKEN }, { status: 200 }),
        ),
    );
    Object.assign(navigator, { clipboard: { writeText } });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('is disabled until a conversation exists', async () => {
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');

    render(<ShareConversationButton conversationId={null} />);

    expect(
      screen.getByRole('button', { name: 'Сподели разговор' }),
    ).toBeDisabled();
  });

  it('creates a share and copies its absolute URL', async () => {
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(screen.getByRole('button', { name: 'Сподели разговор' }));

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/chat/conversation-1/share', {
        method: 'POST',
      });
      expect(writeText).toHaveBeenCalledWith(
        `http://localhost:3000/share/${SHARE_TOKEN}`,
      );
    });

    expect(
      screen.getByRole('button', { name: 'Врската е копирана' }),
    ).toBeEnabled();
  });

  it('announces a failed share request', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 500 }));
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(screen.getByRole('button', { name: 'Сподели разговор' }));

    await expect(
      screen.findByRole('button', { name: 'Споделувањето не успеа' }),
    ).resolves.toBeEnabled();
    expect(writeText).not.toHaveBeenCalled();
  });
});

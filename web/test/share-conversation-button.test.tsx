import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const SHARE_LABEL = 'Сподели разговор';
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

    expect(screen.getByRole('button', { name: SHARE_LABEL })).toBeDisabled();
  });

  it('creates a share and copies its absolute URL', async () => {
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(screen.getByRole('button', { name: SHARE_LABEL }));

    await waitFor(() => {
      const fetchCall = vi.mocked(fetch).mock.calls[0];

      expect(fetchCall?.[0]).toBe('/api/chat/conversation-1/share');
      expect(fetchCall?.[1]?.method).toBe('POST');
      expect(fetchCall?.[1]?.signal).toBeInstanceOf(AbortSignal);
      expect(writeText).toHaveBeenCalledWith(
        `http://localhost:3000/share/${SHARE_TOKEN}`,
      );
    });

    expect(
      screen.getByRole('button', { name: 'Врската е копирана' }),
    ).toBeEnabled();
  });

  it('discards a pending share when the active conversation changes', async () => {
    let resolveShareRequest: ((response: Response) => void) | undefined;
    const pendingResponse = new Promise<Response>((resolve) => {
      resolveShareRequest = resolve;
    });
    vi.mocked(fetch).mockReturnValueOnce(pendingResponse);
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    const { rerender } = render(
      <ShareConversationButton conversationId="conversation-1" />,
    );

    fireEvent.click(screen.getByRole('button', { name: SHARE_LABEL }));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledOnce();
    });
    rerender(<ShareConversationButton conversationId="conversation-2" />);

    const resolveRequest = resolveShareRequest;
    if (resolveRequest === undefined) {
      throw new TypeError('share request did not start');
    }
    await act(() => {
      resolveRequest(
        Response.json({ shareToken: SHARE_TOKEN }, { status: 200 }),
      );
      return Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: SHARE_LABEL })).toBeEnabled();
    });

    expect(writeText).not.toHaveBeenCalled();
  });

  it('announces a failed share request', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 500 }));
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(screen.getByRole('button', { name: SHARE_LABEL }));

    await expect(
      screen.findByRole('button', { name: 'Споделувањето не успеа' }),
    ).resolves.toBeEnabled();
    expect(writeText).not.toHaveBeenCalled();
  });
});

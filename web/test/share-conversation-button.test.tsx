import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const SHARE_LABEL = 'Сподели разговор';
const COPY_LINK_LABEL = 'Копирај ја врската';
const STOP_SHARING_LABEL = 'Прекини споделување';
const SHARE_TOKEN = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d23';
const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue();
const fetchMock = vi.fn<typeof fetch>();
const fetchCallFor = (method: 'DELETE' | 'POST') =>
  fetchMock.mock.calls.find(([, init]) => init?.method === method);

const setupSharingTests = () => {
  writeText.mockClear();
  fetchMock.mockReset();
  fetchMock.mockImplementation((_input, init) =>
    init?.method === 'GET'
      ? Promise.resolve(new Response(null, { status: 204 }))
      : Promise.resolve(
          Response.json({ shareToken: SHARE_TOKEN }, { status: 200 }),
        ),
  );
  vi.stubGlobal('fetch', fetchMock);
  Object.assign(navigator, { clipboard: { writeText } });
};

const teardownSharingTests = () => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
};

describe('ShareConversationButton', () => {
  beforeEach(setupSharingTests);

  afterEach(teardownSharingTests);

  it('is disabled until a conversation exists', async () => {
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');

    render(<ShareConversationButton conversationId={null} />);

    expect(screen.getByRole('button', { name: SHARE_LABEL })).toBeDisabled();
  });

  it('matches the header spacing for active share controls', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ shareToken: SHARE_TOKEN }, { status: 200 }),
    );
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    const copyButton = await screen.findByRole('button', {
      name: COPY_LINK_LABEL,
    });
    const activeControls = copyButton.closest('[aria-live="polite"]');

    if (activeControls === null) {
      throw new TypeError('active share controls were not rendered');
    }

    expect(activeControls).toHaveClass('gap-2');
  });

  it('creates a share and copies its absolute URL', async () => {
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(await screen.findByRole('button', { name: SHARE_LABEL }));

    await waitFor(() => {
      const fetchCall = fetchCallFor('POST');

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
    await expect(
      screen.findByRole(
        'button',
        { name: STOP_SHARING_LABEL },
        { timeout: 2_000 },
      ),
    ).resolves.toBeEnabled();
  });

  it('restores and copies an existing share on initial mount', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ shareToken: SHARE_TOKEN }, { status: 200 }),
    );
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(
      await screen.findByRole('button', { name: COPY_LINK_LABEL }),
    );

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(
        `http://localhost:3000/share/${SHARE_TOKEN}`,
      );
    });

    expect(
      screen.getByRole('button', { name: STOP_SHARING_LABEL }),
    ).toBeEnabled();
  });

  it('loads and revokes an existing share', async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ shareToken: SHARE_TOKEN }, { status: 200 }),
    );
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(
      await screen.findByRole('button', { name: STOP_SHARING_LABEL }),
    );

    await waitFor(() => {
      const revokeCall = fetchCallFor('DELETE');

      expect(revokeCall?.[0]).toBe('/api/chat/conversation-1/share');
      expect(revokeCall?.[1]?.signal).toBeInstanceOf(AbortSignal);
    });

    await expect(
      screen.findByRole('button', { name: SHARE_LABEL }),
    ).resolves.toBeEnabled();
  });

  it('discards a pending share when the active conversation changes', async () => {
    let resolveShareRequest: ((response: Response) => void) | undefined;
    const pendingResponse = new Promise<Response>((resolve) => {
      resolveShareRequest = resolve;
    });
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    const { rerender } = render(
      <ShareConversationButton conversationId="conversation-1" />,
    );

    await screen.findByRole('button', { name: SHARE_LABEL });
    fetchMock.mockReturnValueOnce(pendingResponse);
    fireEvent.click(screen.getByRole('button', { name: SHARE_LABEL }));
    await waitFor(() => {
      expect(fetchCallFor('POST')).toBeDefined();
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
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    await screen.findByRole('button', { name: SHARE_LABEL });
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 500 }));
    fireEvent.click(screen.getByRole('button', { name: SHARE_LABEL }));

    await expect(
      screen.findByRole('button', { name: 'Споделувањето не успеа' }),
    ).resolves.toBeEnabled();
    expect(writeText).not.toHaveBeenCalled();
  });
});

describe('ShareConversationButton failure handling', () => {
  beforeEach(setupSharingTests);

  afterEach(teardownSharingTests);

  it('announces failure when a successful share response has no share token', async () => {
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    await screen.findByRole('button', { name: SHARE_LABEL });
    fetchMock.mockResolvedValueOnce(
      Response.json({ unexpected: SHARE_TOKEN }, { status: 200 }),
    );
    fireEvent.click(screen.getByRole('button', { name: SHARE_LABEL }));

    await expect(
      screen.findByRole('button', { name: 'Споделувањето не успеа' }),
    ).resolves.toBeEnabled();
    expect(writeText).not.toHaveBeenCalled();
    expect(
      screen.queryByRole('button', { name: STOP_SHARING_LABEL }),
    ).toBeNull();
  });

  it('keeps the share URL available without claiming it was copied when clipboard writing fails', async () => {
    writeText.mockRejectedValueOnce(new Error('clipboard unavailable'));
    const { ShareConversationButton } =
      await import('@/components/chat/share-conversation-button');
    render(<ShareConversationButton conversationId="conversation-1" />);

    fireEvent.click(await screen.findByRole('button', { name: SHARE_LABEL }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(
        `http://localhost:3000/share/${SHARE_TOKEN}`,
      );
    });

    expect(screen.getByRole('button', { name: COPY_LINK_LABEL })).toBeEnabled();
    expect(
      screen.queryByRole('button', { name: 'Врската е копирана' }),
    ).toBeNull();
    expect(
      screen.getByRole('button', { name: STOP_SHARING_LABEL }),
    ).toBeEnabled();
  });
});

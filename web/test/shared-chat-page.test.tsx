import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';
import type { SharedChatConversation } from '@/lib/chat-sharing-client';

const RESPONSE_ID_FIELD = 'response_id';
const SHARE_TOKEN = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d23';
class MockChatStateRequestError extends Error {
  readonly status: number;

  constructor(status: number, options?: ErrorOptions) {
    super('Chat state request failed', options);
    this.name = 'MockChatStateRequestError';
    this.status = status;
  }
}
const loadSharedConversation =
  vi.fn<
    (input: { readonly shareToken: string }) => Promise<SharedChatConversation>
  >();
const notFound = vi.fn<() => never>(() => {
  throw new Error('NEXT_NOT_FOUND');
});

vi.mock('next/navigation', () => ({ notFound }));
vi.mock('@/lib/chat-state-client', () => ({
  ChatStateRequestError: MockChatStateRequestError,
}));
vi.mock('@/lib/chat-sharing-client', () => ({
  createChatSharingClient: () => ({ loadSharedConversation }),
}));
vi.mock('@/components/chat/shared-chat-screen', () => ({
  SharedChatScreen: ({
    messages,
    title,
  }: {
    readonly messages: readonly MyUIMessage[];
    readonly title: string;
  }) => (
    <main>
      <h1>{title}</h1>
      <p>
        {messages[0]?.parts[0]?.type === 'text'
          ? messages[0].parts[0].text
          : ''}
      </p>
    </main>
  ),
}));

describe('/share/[token]', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders a sanitized shared conversation without authentication', async () => {
    loadSharedConversation.mockResolvedValueOnce({
      conversation: {
        title: 'Shared enrollment chat',
      },
      messages: [
        {
          content: 'What are the enrollment requirements?',
          id: 'message-1',
          metadata: { responseId: 'private-response-id' },
          parts: null,
          [RESPONSE_ID_FIELD]: null,
          role: 'user',
        },
      ],
    });
    const { default: SharedChatPage } =
      await import('@/app/share/[token]/page');

    render(
      await SharedChatPage({ params: Promise.resolve({ token: SHARE_TOKEN }) }),
    );

    expect(
      screen.getByRole('heading', { name: 'Shared enrollment chat' }),
    ).toBeVisible();
    expect(
      screen.getByText('What are the enrollment requirements?'),
    ).toBeVisible();
    expect(loadSharedConversation).toHaveBeenCalledWith({
      shareToken: SHARE_TOKEN,
    });
  });

  it('renders not found for an unknown share token', async () => {
    loadSharedConversation.mockRejectedValueOnce(
      new MockChatStateRequestError(404),
    );
    const { default: SharedChatPage } =
      await import('@/app/share/[token]/page');

    await expect(
      SharedChatPage({ params: Promise.resolve({ token: SHARE_TOKEN }) }),
    ).rejects.toThrow('NEXT_NOT_FOUND');
    expect(notFound).toHaveBeenCalledOnce();
  });
});

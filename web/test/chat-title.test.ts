import { afterEach, describe, expect, it, vi } from 'vitest';

import { generateChatTitle } from '@/lib/chat-title';

const titleInput = {
  conversationId: '00000000-0000-4000-8000-000000000123',
  messages: [
    {
      id: 'm1',
      parts: [{ text: 'Кога е испитот?', type: 'text' as const }],
      role: 'user' as const,
    },
  ],
  queryTransformModel: 'claude-sonnet-5-5',
};

describe('generateChatTitle', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns null when the title response is malformed JSON', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response('not json', {
        headers: { 'content-type': 'application/json' },
        status: 200,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(generateChatTitle(titleInput)).resolves.toBeNull();
    expect(
      new Headers(fetchMock.mock.calls[0]?.[1]?.headers).get(
        'x-conversation-id',
      ),
    ).toBe(titleInput.conversationId);
  });

  it('returns null when the title response has no usable title', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>().mockResolvedValue(
        Response.json(
          { title: ' '.repeat(3) },
          {
            headers: { 'content-type': 'application/json' },
            status: 200,
          },
        ),
      ),
    );

    await expect(generateChatTitle(titleInput)).resolves.toBeNull();
  });
});

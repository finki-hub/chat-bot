import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

describe('shared chat history', () => {
  it('keeps visible content and sources while removing private metadata', async () => {
    const { sanitizeSharedMessages } = await import('@/lib/chat-history');
    const messages: readonly MyUIMessage[] = [
      {
        id: 'assistant-1',
        metadata: {
          diagnostics: { serverTotalMs: 1_500 },
          feedback: 'like',
          inferenceModel: 'claude-sonnet-5',
          responseId: 'response-1',
          sources: [
            {
              id: 'source-1',
              kind: 'faq',
              snippet: 'Enrollment details',
              title: 'Enrollment',
            },
          ],
          timing: { totalMs: 1_500, ttftMs: 200 },
        },
        parts: [
          { state: 'done', text: 'Visible reasoning', type: 'reasoning' },
          { text: 'Visible answer', type: 'text' },
        ],
        role: 'assistant',
      },
    ];

    const sanitized = sanitizeSharedMessages(messages);

    expect(sanitized).toStrictEqual([
      {
        id: 'assistant-1',
        metadata: {
          sources: [
            {
              id: 'source-1',
              kind: 'faq',
              snippet: 'Enrollment details',
              title: 'Enrollment',
            },
          ],
        },
        parts: [
          { state: 'done', text: 'Visible reasoning', type: 'reasoning' },
          { text: 'Visible answer', type: 'text' },
        ],
        role: 'assistant',
      },
    ]);
  });
});

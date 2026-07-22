import { describe, expect, it } from 'vitest';

import { persistedTurns } from '@/lib/chat-route-state';

describe('persistedTurns', () => {
  it('ends regeneration history at the latest user when a stale assistant precedes the target', () => {
    // Given persisted state with an earlier assistant immediately before the target.
    const messages = [
      { content: 'Current question', id: 'user-1', role: 'user' },
      { content: 'Stale answer', id: 'assistant-stale', role: 'assistant' },
      { content: 'Target answer', id: 'assistant-target', role: 'assistant' },
    ];

    // When history is prepared for regenerating the target assistant.
    const turns = persistedTurns(messages, 'assistant-target', 1);

    // Then the API payload still ends with the user turn that drives regeneration.
    expect(turns).toStrictEqual([
      { content: 'Current question', role: 'user' },
    ]);
  });

  it('rejects regeneration when the target has no prior user turn', () => {
    // Given malformed persisted state containing only the target assistant.
    const messages = [
      { content: 'Orphan answer', id: 'assistant-target', role: 'assistant' },
    ];

    // When history is prepared for regenerating the orphan target.
    const prepareHistory = () =>
      persistedTurns(messages, 'assistant-target', 1);

    // Then the BFF stops before forwarding an empty payload to FastAPI.
    expect(prepareHistory).toThrow(
      'Regeneration history requires a prior user turn',
    );
  });
});

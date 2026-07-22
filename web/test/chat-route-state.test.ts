import { describe, expect, it } from 'vitest';

import {
  persistedTurns,
  retainedServerMessageIdsForRegeneration,
} from '@/lib/chat-route-state';

const ASSISTANT_ROLE = 'assistant';
const TARGET_ASSISTANT_ID = 'assistant-target';
const USER_ROLE = 'user';

describe('persistedTurns', () => {
  it('ends regeneration history at the latest user when a stale assistant precedes the target', () => {
    // Given persisted state with an earlier assistant immediately before the target.
    const messages = [
      { content: 'Current question', id: 'user-1', role: USER_ROLE },
      { content: 'Stale answer', id: 'assistant-stale', role: ASSISTANT_ROLE },
      {
        content: 'Target answer',
        id: TARGET_ASSISTANT_ID,
        role: ASSISTANT_ROLE,
      },
    ];

    // When history is prepared for regenerating the target assistant.
    const turns = persistedTurns(messages, TARGET_ASSISTANT_ID, 1);

    // Then the API payload still ends with the user turn that drives regeneration.
    expect(turns).toStrictEqual([
      { content: 'Current question', role: 'user' },
    ]);
  });

  it('rejects regeneration when the target has no prior user turn', () => {
    // Given malformed persisted state containing only the target assistant.
    const messages = [
      {
        content: 'Orphan answer',
        id: TARGET_ASSISTANT_ID,
        role: ASSISTANT_ROLE,
      },
    ];

    // When history is prepared for regenerating the orphan target.
    const prepareHistory = () =>
      persistedTurns(messages, TARGET_ASSISTANT_ID, 1);

    // Then the BFF stops before forwarding an empty payload to FastAPI.
    expect(prepareHistory).toThrow(
      'Regeneration history requires a prior user turn',
    );
  });
});

describe('retainedServerMessageIdsForRegeneration', () => {
  it('excludes stale assistants between the latest user and target assistant', () => {
    // Given persisted state with a stale assistant immediately before the target.
    const messages = [
      { content: 'Earlier question', id: 'user-1', role: USER_ROLE },
      { content: 'Earlier answer', id: 'assistant-1', role: ASSISTANT_ROLE },
      { content: 'Current question', id: 'user-2', role: USER_ROLE },
      { content: 'Stale answer', id: 'assistant-stale', role: ASSISTANT_ROLE },
      {
        content: 'Target answer',
        id: TARGET_ASSISTANT_ID,
        role: ASSISTANT_ROLE,
      },
    ];

    // When the server-owned replacement window is derived.
    const retainedIds = retainedServerMessageIdsForRegeneration(
      messages,
      TARGET_ASSISTANT_ID,
    );

    // Then valid history, the latest user, and the target remain without the stale answer.
    expect(retainedIds).toStrictEqual([
      'user-1',
      'assistant-1',
      'user-2',
      TARGET_ASSISTANT_ID,
    ]);
  });

  it('rejects a persisted user message as the regeneration target', () => {
    // Given persisted state containing a user ID supplied as the target.
    const messages = [
      { content: 'Earlier question', id: 'user-1', role: USER_ROLE },
      { content: 'Earlier answer', id: 'assistant-1', role: ASSISTANT_ROLE },
      { content: 'Current question', id: 'user-2', role: USER_ROLE },
    ];

    // When the target is validated against the persisted role.
    const retainedIds = retainedServerMessageIdsForRegeneration(
      messages,
      'user-2',
    );

    // Then no replacement window is accepted.
    expect(retainedIds).toBeNull();
  });
});

import { beforeEach, describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import {
  createConversation,
  db,
  deleteConversation,
  listConversations,
  loadMessages,
  renameConversation,
  saveMessages,
  setMessageFeedback,
} from '@/lib/db';

const UUID_LIKE = /[0-9a-f-]{36}/u;

const userMsg = (id: string, text: string): MyUIMessage => ({
  id,
  parts: [{ text, type: 'text' }],
  role: 'user',
});

const assistantMsg = (
  id: string,
  text: string,
  responseId?: string,
): MyUIMessage => ({
  id,
  parts: [{ text, type: 'text' }],
  role: 'assistant',
  ...(responseId && { metadata: { responseId } }),
});

beforeEach(async () => {
  await db.delete();
  await db.open();
});

describe('conversations', () => {
  it('creates a conversation with a generated id and timestamps', async () => {
    const row = await createConversation({
      model: 'claude-sonnet-4-6',
      title: 'Здраво',
    });

    expect(row.id).toMatch(UUID_LIKE);
    expect(row.title).toBe('Здраво');
    expect(row.model).toBe('claude-sonnet-4-6');
    expect(row.createdAt).toBeGreaterThan(0);
    expect(row.updatedAt).toBe(row.createdAt);
  });

  it('honours an explicit id', async () => {
    const row = await createConversation({
      id: 'c-fixed',
      model: 'm',
      title: 'X',
    });

    expect(row.id).toBe('c-fixed');
  });

  it('lists conversations newest-updated first', async () => {
    const a = await createConversation({ id: 'a', model: 'm', title: 'A' });
    const b = await createConversation({ id: 'b', model: 'm', title: 'B' });

    await renameConversation('a', 'A2');

    const list = await listConversations();

    expect(list.map((c) => c.id)).toStrictEqual(['a', 'b']);
    expect(list[0]?.title).toBe('A2');
    expect(a.id).toBe('a');
    expect(b.id).toBe('b');
  });
});

describe('messages', () => {
  it('round-trips UIMessage parts and metadata, ordered by createdAt', async () => {
    await createConversation({ id: 'c1', model: 'm', title: 'C1' });
    await saveMessages('c1', [
      userMsg('m1', 'прашање'),
      assistantMsg('m2', 'одговор', 'resp-123'),
    ]);

    const rows = await loadMessages('c1');

    expect(rows.map((r) => r.id)).toStrictEqual(['m1', 'm2']);
    expect(rows[0]?.parts).toStrictEqual([{ text: 'прашање', type: 'text' }]);
    expect(rows[1]?.metadata).toStrictEqual({ responseId: 'resp-123' });
  });

  it('upserts on re-save (same id) and bumps the conversation updatedAt', async () => {
    const conv = await createConversation({
      id: 'c2',
      model: 'm',
      title: 'C2',
    });

    await saveMessages('c2', [assistantMsg('m1', 'прв')]);
    await saveMessages('c2', [assistantMsg('m1', 'втор')]);

    const rows = await loadMessages('c2');

    expect(rows).toHaveLength(1);
    expect(rows[0]?.parts).toStrictEqual([{ text: 'втор', type: 'text' }]);

    const list = await listConversations();
    const updated = list.find((c) => c.id === 'c2');

    expect(updated?.updatedAt).toBeGreaterThanOrEqual(conv.updatedAt);
  });

  it('persists feedback into the message metadata, preserving prior fields', async () => {
    await createConversation({ id: 'c4', model: 'm', title: 'C4' });
    await saveMessages('c4', [assistantMsg('m1', 'одговор', 'resp-123')]);

    await setMessageFeedback('m1', 'like');

    const rows = await loadMessages('c4');

    expect(rows[0]?.metadata).toStrictEqual({
      feedback: 'like',
      responseId: 'resp-123',
    });
  });

  it('ignores feedback for an unknown message id', async () => {
    await expect(
      setMessageFeedback('missing', 'dislike'),
    ).resolves.toBeUndefined();
  });

  it('deletes a conversation and its messages', async () => {
    await createConversation({ id: 'c3', model: 'm', title: 'C3' });
    await saveMessages('c3', [userMsg('m1', 'x')]);
    await deleteConversation('c3');

    await expect(listConversations()).resolves.toHaveLength(0);
    await expect(loadMessages('c3')).resolves.toHaveLength(0);
  });
});

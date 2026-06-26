import { Dexie, type EntityTable } from 'dexie';

import type { MyUIMessage } from '@/lib/api-types';

export type ConversationRow = {
  createdAt: number;
  id: string;
  model: string;
  title: string;
  updatedAt: number;
};

export type MessageRow = {
  conversationId: string;
  createdAt: number;
  id: string;
  metadata?: NonNullable<MyUIMessage['metadata']>;
  parts: MyUIMessage['parts'];
  role: MyUIMessage['role'];
};

type ChatDb = Dexie & {
  conversations: EntityTable<ConversationRow, 'id'>;
  messages: EntityTable<MessageRow, 'id'>;
};

const createDb = (): ChatDb => {
  const instance = new Dexie('finkiHubChat') as ChatDb;

  instance.version(1).stores({
    conversations: 'id, updatedAt',
    messages: 'id, conversationId, createdAt',
  });

  return instance;
};

export const db = createDb();

const nextNow = ((): (() => number) => {
  let last = 0;

  return () => {
    last = Math.max(Date.now(), last + 1);

    return last;
  };
})();

export const createConversation = async (input: {
  id?: string;
  model: string;
  title: string;
}): Promise<ConversationRow> => {
  const now = nextNow();
  const row: ConversationRow = {
    createdAt: now,
    id: input.id ?? crypto.randomUUID(),
    model: input.model,
    title: input.title,
    updatedAt: now,
  };

  await db.conversations.put(row);

  return row;
};

export const listConversations = (): Promise<ConversationRow[]> =>
  db.conversations.orderBy('updatedAt').reverse().toArray();

export const getConversation = (
  id: string,
): Promise<ConversationRow | undefined> => db.conversations.get(id);

export const renameConversation = async (
  id: string,
  title: string,
): Promise<void> => {
  await db.conversations.update(id, { title, updatedAt: nextNow() });
};

export const deleteConversation = async (id: string): Promise<void> => {
  await db.transaction('rw', db.conversations, db.messages, async () => {
    await db.messages.where('conversationId').equals(id).delete();
    await db.conversations.delete(id);
  });
};

export const clearAllConversations = async (): Promise<void> => {
  await db.transaction('rw', db.conversations, db.messages, async () => {
    await db.messages.clear();
    await db.conversations.clear();
  });
};

export const loadMessages = (conversationId: string): Promise<MessageRow[]> =>
  db.messages
    .where('conversationId')
    .equals(conversationId)
    .sortBy('createdAt');

export const saveMessages = async (
  conversationId: string,
  messages: MyUIMessage[],
): Promise<void> => {
  const base = nextNow();
  const rows: MessageRow[] = messages.map((m, i) => ({
    conversationId,
    createdAt: base + i,
    id: m.id,
    metadata: m.metadata,
    parts: m.parts,
    role: m.role,
  }));

  await db.transaction('rw', db.conversations, db.messages, async () => {
    await db.messages.bulkPut(rows);
    await db.conversations.update(conversationId, { updatedAt: nextNow() });
  });
};

export const replaceConversationMessages = async (
  conversationId: string,
  messages: MyUIMessage[],
): Promise<void> => {
  await db.transaction('rw', db.conversations, db.messages, async () => {
    const existingRows = await db.messages
      .where('conversationId')
      .equals(conversationId)
      .toArray();
    const existing = new Map(
      existingRows.map((row) => [row.id, row.createdAt] as const),
    );
    const base = nextNow();
    const rows: MessageRow[] = messages.map((message, index) => ({
      conversationId,
      createdAt: existing.get(message.id) ?? base + index,
      id: message.id,
      metadata: message.metadata,
      parts: message.parts,
      role: message.role,
    }));
    const keptIds = new Set(rows.map((row) => row.id));
    const staleIds: string[] = [];
    for (const id of existing.keys()) {
      if (!keptIds.has(id)) {
        staleIds.push(id);
      }
    }

    if (staleIds.length > 0) {
      await db.messages.bulkDelete(staleIds);
    }

    await db.messages.bulkPut(rows);
    await db.conversations.update(conversationId, { updatedAt: nextNow() });
  });
};

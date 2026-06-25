// Local (IndexedDB) conversation + message store. Stores messages in the AI SDK
// UIMessage shape so metadata.responseId survives reloads (spec §7).
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
    conversations: 'id, updatedAt', // primary key + index for ordered listing
    messages: 'id, conversationId, createdAt', // query by conversation, ordered by time
  });

  return instance;
};

export const db = createDb();

// Monotonic millisecond clock: Date.now() can repeat within the same tick, which
// would make two writes tie on updatedAt and lose their relative order. Advancing
// past the last issued value guarantees a later write always sorts newer.
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
    createdAt: base + i, // preserve send order within one batch
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

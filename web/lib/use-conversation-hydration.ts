'use client';

import { type RefObject, type SetStateAction, useEffect } from 'react';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import {
  createConversation,
  getConversation,
  loadMessages,
  type MessageRow,
  replaceConversationMessages,
} from '@/lib/db';
import { joinText } from '@/lib/message-parts';
import { deriveTitle } from '@/lib/messages';

type UseConversationHydrationOptions = {
  readonly activeId: null | string;
  readonly convoIdRef: RefObject<null | string>;
  readonly model: string;
  readonly setActiveError: (value: ErrorNotice | undefined) => void;
  readonly setActiveId: (id: null | string) => void;
  readonly setActiveStatus: (
    value: undefined | { label: string; tool?: string },
  ) => void;
  readonly setMessages: (messages: SetStateAction<MyUIMessage[]>) => void;
};

const fromRow = (row: MessageRow): MyUIMessage => ({
  id: row.id,
  metadata: row.metadata,
  parts: row.parts,
  role: row.role,
});

const mergeCurrentWithLocal = (
  current: MyUIMessage[],
  localMessages: MyUIMessage[],
): MyUIMessage[] => {
  const currentIds = new Set(current.map((message) => message.id));
  const hasSameLocalBase =
    localMessages.length > 0 &&
    localMessages.every((message) => currentIds.has(message.id));

  if (hasSameLocalBase && current.length > localMessages.length) {
    return current;
  }

  const localIds = new Set(localMessages.map((message) => message.id));
  const extraCurrentMessages = current.filter(
    (message) => !localIds.has(message.id),
  );

  if (!(hasSameLocalBase && extraCurrentMessages.length > 0)) {
    return localMessages;
  }

  return [...localMessages, ...extraCurrentMessages];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isUiMessage = (value: unknown): value is MyUIMessage => {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value['id'] === 'string' &&
    isRecord(value['metadata']) &&
    Array.isArray(value['parts']) &&
    (value['role'] === 'assistant' || value['role'] === 'user')
  );
};

type ServerHistory = {
  readonly conversation: {
    readonly id: string;
    readonly model: null | string;
    readonly title: null | string;
  };
  readonly messages: MyUIMessage[];
};

const isServerHistory = (value: unknown): value is ServerHistory => {
  if (!isRecord(value) || !isRecord(value['conversation'])) {
    return false;
  }

  const { conversation } = value;

  return (
    typeof conversation['id'] === 'string' &&
    (conversation['model'] === null ||
      typeof conversation['model'] === 'string') &&
    (conversation['title'] === null ||
      typeof conversation['title'] === 'string') &&
    Array.isArray(value['messages']) &&
    value['messages'].every(isUiMessage)
  );
};

const loadServerHistory = async (id: string): Promise<null | ServerHistory> => {
  let response: Response;

  try {
    response = await fetch(`/api/chat/${encodeURIComponent(id)}/history`, {
      method: 'GET',
    });
  } catch {
    return null;
  }

  if (!response.ok) {
    return null;
  }

  let body: unknown;

  try {
    body = await response.json();
  } catch {
    return null;
  }

  return isServerHistory(body) ? body : null;
};

const titleFromHistory = (history: ServerHistory): string => {
  if (history.conversation.title !== null) {
    return history.conversation.title;
  }

  const firstUserMessage = history.messages.find(
    (message) => message.role === 'user',
  );
  return deriveTitle(
    firstUserMessage === undefined ? '' : joinText(firstUserMessage),
  );
};

export const useConversationHydration = ({
  activeId,
  convoIdRef,
  model,
  setActiveError,
  setActiveId,
  setActiveStatus,
  setMessages,
}: UseConversationHydrationOptions): void => {
  useEffect(() => {
    convoIdRef.current = activeId;
    setActiveError(undefined);
    setActiveStatus(undefined);
    let cancelled = false;
    const isCancelled = (): boolean => cancelled;

    const hydrate = async (id: string): Promise<void> => {
      const convo = await getConversation(id);

      if (convo !== undefined && !isCancelled()) {
        const loaded = await loadMessages(id);
        if (!isCancelled()) {
          const localMessages = loaded.map(fromRow);
          setMessages((current) =>
            mergeCurrentWithLocal(current, localMessages),
          );
        }
        return;
      }

      if (convo !== undefined || isCancelled()) {
        return;
      }

      const serverHistory = await loadServerHistory(id);
      if (serverHistory !== null) {
        if (!isCancelled()) {
          const serverMessages = serverHistory.messages;
          setMessages(serverMessages);
          await createConversation({
            id,
            model: serverHistory.conversation.model ?? model,
            title: titleFromHistory(serverHistory),
          });
          await replaceConversationMessages(id, serverMessages);
        }
        return;
      }

      if (!isCancelled()) {
        setActiveId(null);
        setMessages([]);
      }
    };

    if (activeId) {
      fireAndForget(hydrate(activeId));
    } else {
      setMessages([]);
    }

    return () => {
      cancelled = true;
    };
  }, [
    activeId,
    convoIdRef,
    model,
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  ]);
};

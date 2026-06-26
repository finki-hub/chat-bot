'use client';

import { type RefObject, useEffect } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { getConversation, loadMessages, type MessageRow } from '@/lib/db';

type UseConversationHydrationOptions = {
  readonly activeId: null | string;
  readonly convoIdRef: RefObject<null | string>;
  readonly setActiveError: (
    value: undefined | { code: string; message: string },
  ) => void;
  readonly setActiveId: (id: null | string) => void;
  readonly setActiveStatus: (
    value: undefined | { label: string; tool?: string },
  ) => void;
  readonly setMessages: (messages: MyUIMessage[]) => void;
};

const fromRow = (row: MessageRow): MyUIMessage => ({
  id: row.id,
  metadata: row.metadata,
  parts: row.parts,
  role: row.role,
});

export const useConversationHydration = ({
  activeId,
  convoIdRef,
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

      if (!isCancelled() && convo === undefined) {
        setActiveId(null);
        setMessages([]);
        return;
      }

      if (convo === undefined || isCancelled()) {
        return;
      }

      const loaded = await loadMessages(id);
      if (!isCancelled()) {
        setMessages(loaded.map(fromRow));
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
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  ]);
};

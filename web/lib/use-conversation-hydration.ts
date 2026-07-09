'use client';

import { type RefObject, type SetStateAction, useEffect } from 'react';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { loadChatConversationHistory } from '@/lib/transport';

type UseConversationHydrationOptions = {
  readonly activeId: null | string;
  readonly convoIdRef: RefObject<null | string>;
  readonly setActiveError: (value: ErrorNotice | undefined) => void;
  readonly setActiveId: (id: null | string) => void;
  readonly setActiveStatus: (
    value: undefined | { label: string; tool?: string },
  ) => void;
  readonly setMessages: (messages: SetStateAction<MyUIMessage[]>) => void;
};

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
      const serverHistory = await loadChatConversationHistory(id);
      if (serverHistory !== null) {
        if (!isCancelled()) {
          setMessages((current) =>
            serverHistory.messages.length === 0 && current.length > 0
              ? current
              : [...serverHistory.messages],
          );
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
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  ]);
};

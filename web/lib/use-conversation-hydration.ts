'use client';

import {
  type RefObject,
  type SetStateAction,
  useEffect,
  useLayoutEffect,
  useState,
} from 'react';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { t } from '@/lib/i18n';
import {
  ChatConversationRequestError,
  loadChatConversationHistory,
} from '@/lib/transport';

type UseConversationHydrationOptions = {
  readonly activeId: null | string;
  readonly activeStreamConversationIdRef: RefObject<null | string>;
  readonly convoIdRef: RefObject<null | string>;
  readonly preserveEmptyHydrationIdRef: RefObject<null | string>;
  readonly setActiveError: (value: ErrorNotice | undefined) => void;
  readonly setActiveId: (id: null | string) => void;
  readonly setActiveStatus: (
    value: undefined | { label: string; tool?: string },
  ) => void;
  readonly setMessages: (messages: SetStateAction<MyUIMessage[]>) => void;
};

export const useConversationHydration = ({
  activeId,
  activeStreamConversationIdRef,
  convoIdRef,
  preserveEmptyHydrationIdRef,
  setActiveError,
  setActiveId,
  setActiveStatus,
  setMessages,
}: UseConversationHydrationOptions): boolean => {
  const [hydratingId, setHydratingId] = useState<null | string>(null);

  useLayoutEffect(() => {
    setHydratingId(activeId);
  }, [activeId]);

  useEffect(() => {
    convoIdRef.current = activeId;
    setActiveError(undefined);
    setActiveStatus(undefined);
    let cancelled = false;
    const isCancelled = (): boolean => cancelled;
    const clearPreserveMarker = (id: string): void => {
      if (preserveEmptyHydrationIdRef.current === id) {
        preserveEmptyHydrationIdRef.current = null;
      }
    };
    const clearActiveStreamMarker = (id: string): void => {
      if (activeStreamConversationIdRef.current === id) {
        activeStreamConversationIdRef.current = null;
      }
    };
    const hasLocalConversationState = (id: string): boolean =>
      preserveEmptyHydrationIdRef.current === id ||
      activeStreamConversationIdRef.current === id;

    // eslint-disable-next-line sonarjs/cognitive-complexity -- cancellation and server/local reconciliation are one request state machine
    const hydrate = async (id: string): Promise<void> => {
      try {
        const serverHistory = await loadChatConversationHistory(id);
        if (serverHistory !== null) {
          if (!isCancelled()) {
            setMessages((current) =>
              serverHistory.messages.length === 0 &&
              current.length > 0 &&
              hasLocalConversationState(id)
                ? current
                : [...serverHistory.messages],
            );
            clearPreserveMarker(id);
            clearActiveStreamMarker(id);
          }
          return;
        }

        if (!isCancelled()) {
          clearPreserveMarker(id);
          clearActiveStreamMarker(id);
          setActiveError({
            code: 'history_load',
            message: t('conversation.historyLoadError'),
          });
        }
      } catch (error) {
        if (!isCancelled()) {
          if (
            error instanceof ChatConversationRequestError &&
            error.status === 404
          ) {
            if (hasLocalConversationState(id)) {
              return;
            }
            clearPreserveMarker(id);
            clearActiveStreamMarker(id);
            setActiveId(null);
            setMessages([]);
            return;
          }
          clearPreserveMarker(id);
          clearActiveStreamMarker(id);
          if (
            error instanceof ChatConversationRequestError ||
            error instanceof TypeError
          ) {
            setActiveError({
              code: 'history_load',
              message: t('conversation.historyLoadError'),
            });
            return;
          }
          throw error;
        }
      } finally {
        if (!isCancelled()) {
          setHydratingId((current) => (current === id ? null : current));
        }
      }
    };

    if (activeId) {
      fireAndForget(hydrate(activeId));
    } else {
      preserveEmptyHydrationIdRef.current = null;
      setHydratingId(null);
      setMessages([]);
    }

    return () => {
      cancelled = true;
    };
  }, [
    activeStreamConversationIdRef,
    activeId,
    convoIdRef,
    preserveEmptyHydrationIdRef,
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  ]);

  return activeId !== null && hydratingId === activeId;
};

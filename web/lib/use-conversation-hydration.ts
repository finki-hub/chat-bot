'use client';

import {
  type Dispatch,
  type RefObject,
  type SetStateAction,
  useCallback,
  useEffect,
  useLayoutEffect,
  useState,
} from 'react';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { reconcileHydratedMessages } from '@/lib/conversation-message-state';
import { t } from '@/lib/i18n';
import {
  ChatConversationRequestError,
  loadChatConversationHistory,
} from '@/lib/transport';

type ConversationHydration = {
  readonly hydratingConversation: boolean;
  readonly retryHydration: () => void;
};

type UseConversationHydrationOptions = {
  readonly activeId: null | string;
  readonly activeStreamConversationIdRef: RefObject<null | string>;
  readonly convoIdRef: RefObject<null | string>;
  readonly preserveEmptyHydrationIdRef: RefObject<null | string>;
  readonly setActiveError: Dispatch<SetStateAction<ErrorNotice | undefined>>;
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
}: UseConversationHydrationOptions): ConversationHydration => {
  const [hydratingId, setHydratingId] = useState<null | string>(null);
  const [hydrationAttempt, setHydrationAttempt] = useState(0);

  const retryHydration = useCallback(() => {
    if (activeId === null || hydratingId === activeId) {
      return;
    }
    setHydratingId(activeId);
    setHydrationAttempt((current) => current + 1);
  }, [activeId, hydratingId]);

  useLayoutEffect(() => {
    setHydratingId(activeId);
    setActiveError(undefined);
  }, [activeId, setActiveError]);

  useEffect(() => {
    convoIdRef.current = activeId;
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
            setActiveError((current) =>
              current?.code === 'history_load' ? undefined : current,
            );
            setMessages((current) => {
              if (
                serverHistory.messages.length === 0 &&
                current.length > 0 &&
                hasLocalConversationState(id)
              ) {
                return current;
              }
              return reconcileHydratedMessages({
                activeStream: serverHistory.conversation.activeStream,
                current,
                persisted: serverHistory.messages,
              });
            });
            clearPreserveMarker(id);
            clearActiveStreamMarker(id);
          }
          return;
        }

        if (!isCancelled()) {
          setActiveError((current) =>
            current === undefined || current.code === 'history_load'
              ? {
                  code: 'history_load',
                  message: t('conversation.historyLoadError'),
                }
              : current,
          );
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
          if (
            error instanceof ChatConversationRequestError ||
            error instanceof TypeError
          ) {
            setActiveError((current) =>
              current === undefined || current.code === 'history_load'
                ? {
                    code: 'history_load',
                    message: t('conversation.historyLoadError'),
                  }
                : current,
            );
            return;
          }
          clearPreserveMarker(id);
          clearActiveStreamMarker(id);
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
    hydrationAttempt,
    preserveEmptyHydrationIdRef,
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  ]);

  return {
    hydratingConversation: activeId !== null && hydratingId === activeId,
    retryHydration,
  };
};

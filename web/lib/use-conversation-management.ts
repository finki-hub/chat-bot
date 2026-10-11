import {
  type Dispatch,
  type RefObject,
  type SetStateAction,
  useCallback,
} from 'react';
import { flushSync } from 'react-dom';

import type { ErrorNotice, MyUIMessage } from '@/lib/api-types';
import type { StopOrder } from '@/lib/use-stop-chat';

import { fireAndForget } from '@/lib/async';
import { t } from '@/lib/i18n';
import { deriveTitle } from '@/lib/messages';
import {
  ChatConversationRequestError,
  clearChatConversations,
  deleteChatConversation,
  DeleteChatConversationError,
  saveChatConversation,
  StopChatStreamError,
} from '@/lib/transport';

type UseConversationManagementOptions = {
  readonly applyGeneratedTitle: (
    id: string,
    messages: readonly MyUIMessage[],
    expectedTitle: string,
  ) => Promise<void>;
  readonly convoIdRef: RefObject<null | string>;
  readonly handleStop: (order?: StopOrder) => Promise<void>;
  readonly model: string;
  readonly preserveEmptyHydrationIdRef: RefObject<null | string>;
  readonly refreshConversations: () => Promise<void>;
  readonly sendMessageRef: RefObject<(message: MyUIMessage) => Promise<void>>;
  readonly setActiveError: Dispatch<SetStateAction<ErrorNotice | undefined>>;
  readonly setActiveId: (id: null | string) => void;
  readonly setMessages: Dispatch<SetStateAction<MyUIMessage[]>>;
  readonly status: string;
};

export const useConversationManagement = ({
  applyGeneratedTitle,
  convoIdRef,
  handleStop,
  model,
  preserveEmptyHydrationIdRef,
  refreshConversations,
  sendMessageRef,
  setActiveError,
  setActiveId,
  setMessages,
  status,
}: UseConversationManagementOptions) => {
  const resetActiveConversation = useCallback(() => {
    preserveEmptyHydrationIdRef.current = null;
    setActiveId(null);
    setMessages([]);
    setActiveError(undefined);
    convoIdRef.current = null;
  }, [
    convoIdRef,
    preserveEmptyHydrationIdRef,
    setActiveError,
    setActiveId,
    setMessages,
  ]);

  const handleNewChat = useCallback(() => {
    if (status !== 'ready') {
      fireAndForget(
        (async () => {
          await handleStop('local-first');
          resetActiveConversation();
        })(),
      );
      return;
    }

    resetActiveConversation();
  }, [handleStop, resetActiveConversation, status]);

  const handleSubmit = useCallback(
    async (text: string): Promise<boolean> => {
      setActiveError(undefined);
      const existing = convoIdRef.current;
      let cid = existing;
      let expectedTitle: null | string = null;
      if (!existing) {
        expectedTitle = deriveTitle(text);
        cid = crypto.randomUUID();
        try {
          await saveChatConversation({
            id: cid,
            model,
            title: expectedTitle,
          });
        } catch (error) {
          if (
            error instanceof ChatConversationRequestError ||
            error instanceof TypeError
          ) {
            setActiveError({
              code: 'conversation_create',
              message: t('conversation.createError'),
            });
            return false;
          }
          throw error;
        }
        // eslint-disable-next-line require-atomic-updates -- fresh id, not a read-modify-write race
        convoIdRef.current = cid;
        preserveEmptyHydrationIdRef.current = cid;
        flushSync(() => {
          setActiveId(cid);
        });
        fireAndForget(refreshConversations());
      }
      if (!cid) {
        return false;
      }
      const userMessage: MyUIMessage = {
        id: crypto.randomUUID(),
        metadata: {},
        parts: [{ text, type: 'text' }],
        role: 'user',
      };
      if (expectedTitle !== null) {
        fireAndForget(applyGeneratedTitle(cid, [userMessage], expectedTitle));
      }
      fireAndForget(sendMessageRef.current(userMessage));
      return true;
    },
    [
      applyGeneratedTitle,
      convoIdRef,
      model,
      preserveEmptyHydrationIdRef,
      refreshConversations,
      sendMessageRef,
      setActiveError,
      setActiveId,
    ],
  );

  const handleSelect = useCallback(
    (id: string) => {
      if (id !== convoIdRef.current && status !== 'ready') {
        fireAndForget(
          (async () => {
            await handleStop('local-first');
            convoIdRef.current = id;
            setActiveId(id);
          })(),
        );
        return;
      }
      convoIdRef.current = id;
      setActiveId(id);
    },
    [convoIdRef, handleStop, setActiveId, status],
  );

  const deleteConversationEverywhere = useCallback(
    async (id: string) => {
      const deletingActiveConversation = convoIdRef.current === id;
      if (deletingActiveConversation && status !== 'ready') {
        await handleStop();
      }

      try {
        await deleteChatConversation(id);
      } catch (error) {
        const serverConversationAlreadyDeleted =
          error instanceof DeleteChatConversationError && error.status === 404;

        if (!serverConversationAlreadyDeleted) {
          throw error;
        }
      }
      if (deletingActiveConversation && convoIdRef.current === id) {
        resetActiveConversation();
      }
      await refreshConversations();
    },
    [
      convoIdRef,
      handleStop,
      refreshConversations,
      resetActiveConversation,
      status,
    ],
  );

  const handleDelete = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        await deleteConversationEverywhere(id);
        return true;
      } catch (error) {
        if (
          error instanceof DeleteChatConversationError ||
          error instanceof StopChatStreamError ||
          error instanceof TypeError
        ) {
          return false;
        }
        throw error;
      }
    },
    [deleteConversationEverywhere],
  );

  const handleClearAll = useCallback(async (): Promise<boolean> => {
    try {
      if (status !== 'ready') {
        await handleStop();
      }
      await clearChatConversations();
      resetActiveConversation();
      await refreshConversations();
      return true;
    } catch (error) {
      if (
        error instanceof ChatConversationRequestError ||
        error instanceof StopChatStreamError ||
        error instanceof TypeError
      ) {
        return false;
      }
      throw error;
    }
  }, [handleStop, refreshConversations, resetActiveConversation, status]);

  const submitMessage = useCallback(
    (text: string): Promise<boolean> => handleSubmit(text),
    [handleSubmit],
  );

  const handleRename = useCallback(
    async (id: string, title: string): Promise<boolean> => {
      try {
        await saveChatConversation({ id, title });
        await refreshConversations();
        return true;
      } catch (error) {
        if (
          error instanceof ChatConversationRequestError ||
          error instanceof TypeError
        ) {
          return false;
        }
        throw error;
      }
    },
    [refreshConversations],
  );

  return {
    handleClearAll,
    handleDelete,
    handleNewChat,
    handleRename,
    handleSelect,
    submitMessage,
  };
};

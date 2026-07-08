'use client';

import { useCallback } from 'react';
import { flushSync } from 'react-dom';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { renderAnswerActions } from '@/lib/conversation-actions';
import {
  applyFeedback,
  previewRegeneration,
} from '@/lib/conversation-message-state';
import {
  clearAllConversations,
  createConversation,
  deleteConversation,
  renameConversation,
  saveMessages,
  setMessageFeedback,
} from '@/lib/db';
import { deriveTitle } from '@/lib/messages';
import { deleteChatConversation } from '@/lib/transport';
import { useUiStore } from '@/lib/ui-store';
import { useConversationChatRuntime } from '@/lib/use-conversation-chat-runtime';
import { useConversationList } from '@/lib/use-conversation-list';
import { useGeneratedTitle } from '@/lib/use-generated-title';
import { useStopChat } from '@/lib/use-stop-chat';

export const useConversations = (
  model: string,
  disabled = false,
  reasoning = false,
) => {
  const activeId = useUiStore((s) => s.activeConversationId);
  const setActiveId = useUiStore((s) => s.setActiveConversationId);

  const { conversations, refreshConversations } = useConversationList();
  const {
    activeError,
    activeStatus,
    convoIdRef,
    messages,
    modelRef,
    regenerate,
    regeneratingMessageId,
    regeneratingMessageIdRef,
    sendMessageRef,
    setActiveError,
    setActiveStatus,
    setMessages,
    setRegeneratingMessageId,
    status,
    stop,
  } = useConversationChatRuntime({
    activeId,
    model,
    reasoning,
    refreshConversations,
    setActiveId,
  });
  const { applyGeneratedTitle, generatingTitleId, handleGenerateTitle } =
    useGeneratedTitle({ conversations, modelRef, refreshConversations });
  const handleStop = useStopChat({ convoIdRef, messages, model, stop });

  const handleNewChat = useCallback(() => {
    if (status !== 'ready') {
      fireAndForget(
        (async () => {
          await handleStop('local-first');
          setActiveId(null);
          setMessages([]);
          setActiveError(undefined);
          convoIdRef.current = null;
        })(),
      );
      return;
    }
    setActiveId(null);
    setMessages([]);
    setActiveError(undefined);
    convoIdRef.current = null;
  }, [
    convoIdRef,
    handleStop,
    setActiveError,
    setActiveId,
    setMessages,
    status,
  ]);

  const handleSubmit = useCallback(
    async (text: string) => {
      setActiveError(undefined);
      const existing = convoIdRef.current;
      let cid = existing;
      let expectedTitle: null | string = null;
      if (!existing) {
        expectedTitle = deriveTitle(text);
        const convo = await createConversation({
          model,
          title: expectedTitle,
        });
        cid = convo.id;
        // eslint-disable-next-line require-atomic-updates -- fresh id, not a read-modify-write race
        convoIdRef.current = convo.id;
        // eslint-disable-next-line @eslint-react/dom-no-flush-sync -- conversation id must be committed before useChat resumes the new stream
        flushSync(() => {
          setActiveId(convo.id);
        });
        await refreshConversations();
      }
      if (!cid) {
        return;
      }
      const userMessage: MyUIMessage = {
        id: crypto.randomUUID(),
        metadata: {},
        parts: [{ text, type: 'text' }],
        role: 'user',
      };
      await saveMessages(cid, [userMessage]);
      if (expectedTitle !== null) {
        fireAndForget(applyGeneratedTitle(cid, [userMessage], expectedTitle));
      }
      fireAndForget(sendMessageRef.current(userMessage));
    },
    [
      applyGeneratedTitle,
      convoIdRef,
      model,
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
            setActiveId(id);
          })(),
        );
        return;
      }
      setActiveId(id);
    },
    [convoIdRef, handleStop, setActiveId, status],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      await deleteChatConversation(id);
      await deleteConversation(id);
      if (convoIdRef.current === id) {
        handleNewChat();
      }
      await refreshConversations();
    },
    [convoIdRef, handleNewChat, refreshConversations],
  );

  const handleClearAll = useCallback(async () => {
    await clearAllConversations();
    handleNewChat();
    await refreshConversations();
  }, [handleNewChat, refreshConversations]);

  const submitMessage = useCallback(
    (text: string) => {
      fireAndForget(handleSubmit(text));
    },
    [handleSubmit],
  );

  const retry = useCallback(() => {
    if (disabled) {
      return;
    }
    setActiveError(undefined);
    fireAndForget(regenerate());
  }, [disabled, regenerate, setActiveError]);

  const regenerateMessage = useCallback(
    (options: { messageId: string }) => {
      if (disabled) {
        return;
      }
      regeneratingMessageIdRef.current = options.messageId;
      setRegeneratingMessageId(options.messageId);
      void regenerate(options);
      setActiveError(undefined);
      setActiveStatus(undefined);
    },
    [
      disabled,
      regenerate,
      regeneratingMessageIdRef,
      setActiveError,
      setActiveStatus,
      setRegeneratingMessageId,
    ],
  );

  const handleRename = useCallback(
    async (id: string, title: string) => {
      await renameConversation(id, title);
      await refreshConversations();
    },
    [refreshConversations],
  );

  const recordFeedback = useCallback(
    (messageId: string, vote: FeedbackType) => {
      setMessages(applyFeedback(messageId, vote));
      fireAndForget(setMessageFeedback(messageId, vote));
    },
    [setMessages],
  );

  const visibleMessages = previewRegeneration(messages, regeneratingMessageId);
  const renderActions = renderAnswerActions({
    disabled,
    messages: visibleMessages,
    onVote: recordFeedback,
    regenerate: regenerateMessage,
    status,
  });

  return {
    activeError,
    activeId,
    activeStatus,
    conversations,
    generatingTitleId,
    messages: visibleMessages,
    onClearAll: handleClearAll,
    onDelete: handleDelete,
    onGenerateTitle: handleGenerateTitle,
    onNewChat: handleNewChat,
    onRename: handleRename,
    onSelect: handleSelect,
    onStop: () => {
      fireAndForget(handleStop());
    },
    renderActions,
    retry,
    status,
    submitMessage,
  };
};

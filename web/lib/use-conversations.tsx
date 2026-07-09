'use client';

import { useCallback } from 'react';

import type { FeedbackType } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { renderAnswerActions } from '@/lib/conversation-actions';
import {
  applyFeedback,
  previewRegeneration,
} from '@/lib/conversation-message-state';
import { useUiStore } from '@/lib/ui-store';
import { useConversationChatRuntime } from '@/lib/use-conversation-chat-runtime';
import { useConversationList } from '@/lib/use-conversation-list';
import { useConversationManagement } from '@/lib/use-conversation-management';
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
  const {
    handleClearAll,
    handleDelete,
    handleNewChat,
    handleRename,
    handleSelect,
    submitMessage,
  } = useConversationManagement({
    applyGeneratedTitle,
    convoIdRef,
    handleStop,
    model,
    refreshConversations,
    sendMessageRef,
    setActiveError,
    setActiveId,
    setMessages,
    status,
  });

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

  const recordFeedback = useCallback(
    (messageId: string, vote: FeedbackType) => {
      setMessages(applyFeedback(messageId, vote));
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

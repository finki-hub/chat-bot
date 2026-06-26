'use client';

import { useChat } from '@ai-sdk/react';
import { useCallback, useRef, useState } from 'react';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { renderAnswerActions } from '@/lib/conversation-actions';
import {
  applyFeedback,
  finalizeMessage,
  previewRegeneration,
  replaceFinishedMessage,
} from '@/lib/conversation-message-state';
import {
  clearAllConversations,
  createConversation,
  deleteConversation,
  renameConversation,
  replaceConversationMessages,
  saveMessages,
  setMessageFeedback,
} from '@/lib/db';
import { deriveTitle } from '@/lib/messages';
import { buildChatTransport } from '@/lib/transport';
import { useUiStore } from '@/lib/ui-store';
import { useConversationHydration } from '@/lib/use-conversation-hydration';
import { useConversationList } from '@/lib/use-conversation-list';
import { useStreamTiming } from '@/lib/use-stream-timing';

export const useConversations = (model: string, disabled = false) => {
  const activeId = useUiStore((s) => s.activeConversationId);
  const setActiveId = useUiStore((s) => s.setActiveConversationId);

  const { conversations, refreshConversations } = useConversationList();
  const [activeStatus, setActiveStatus] = useState<
    undefined | { label: string; tool?: string }
  >();
  const [activeError, setActiveError] = useState<
    undefined | { code: string; message: string }
  >();
  const convoIdRef = useRef<null | string>(activeId);
  const startedAtRef = useRef<null | number>(null);
  const firstTokenAtRef = useRef<null | number>(null);
  const regeneratingMessageIdRef = useRef<null | string>(null);
  const [regeneratingMessageId, setRegeneratingMessageId] = useState<
    null | string
  >(null);
  const [streamStartedAt, setStreamStartedAt] = useState<null | number>(null);

  const persistFinished = useCallback(
    async (allMessages: readonly MyUIMessage[]): Promise<void> => {
      const cid = convoIdRef.current;
      if (!cid) {
        return;
      }
      await replaceConversationMessages(cid, [...allMessages]);
      await refreshConversations();
    },
    [refreshConversations],
  );

  const { messages, regenerate, sendMessage, setMessages, status, stop } =
    useChat<MyUIMessage>({
      onData: (part) => {
        if (part.type === 'data-status') {
          setActiveStatus(part.data);
        } else {
          setActiveError(part.data);
        }
      },
      onError: () => {
        regeneratingMessageIdRef.current = null;
        setRegeneratingMessageId(null);
      },
      onFinish: ({ message }) => {
        setActiveStatus(undefined);
        const replacementId = regeneratingMessageIdRef.current;
        regeneratingMessageIdRef.current = null;
        setRegeneratingMessageId(null);
        const finalizedBase = finalizeMessage(
          message,
          startedAtRef.current,
          firstTokenAtRef.current,
        );
        const finalized =
          replacementId === null
            ? finalizedBase
            : { ...finalizedBase, id: replacementId };
        setMessages((prev) => {
          const next = replaceFinishedMessage({
            pruneAfterReplacement: replacementId !== null,
            replacement: finalized,
            streamMessageId: message.id,
          })(prev);
          fireAndForget(persistFinished(next));
          return next;
        });
      },
      transport: buildChatTransport(() => ({ model })),
    });

  useStreamTiming({
    firstTokenAtRef,
    messages,
    setStreamStartedAt,
    startedAtRef,
    status,
  });

  useConversationHydration({
    activeId,
    convoIdRef,
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  });

  const handleNewChat = useCallback(() => {
    setActiveId(null);
    setMessages([]);
    setActiveError(undefined);
    convoIdRef.current = null;
  }, [setActiveId, setMessages]);

  const handleSubmit = useCallback(
    async (text: string) => {
      setActiveError(undefined);
      const existing = convoIdRef.current;
      let cid = existing;
      if (!existing) {
        const convo = await createConversation({
          model,
          title: deriveTitle(text),
        });
        cid = convo.id;
        // eslint-disable-next-line require-atomic-updates -- fresh id, not a read-modify-write race
        convoIdRef.current = convo.id;
        setActiveId(convo.id);
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
      fireAndForget(sendMessage(userMessage));
    },
    [model, refreshConversations, sendMessage, setActiveId],
  );

  const handleSelect = useCallback(
    (id: string) => {
      setActiveId(id);
    },
    [setActiveId],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      await deleteConversation(id);
      if (convoIdRef.current === id) {
        handleNewChat();
      }
      await refreshConversations();
    },
    [handleNewChat, refreshConversations],
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
    fireAndForget(regenerate());
  }, [disabled, regenerate]);

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
    [disabled, regenerate],
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
    messages: visibleMessages,
    onClearAll: handleClearAll,
    onDelete: handleDelete,
    onNewChat: handleNewChat,
    onRename: handleRename,
    onSelect: handleSelect,
    onStop: stop,
    renderActions,
    retry,
    status,
    streamStartedAt,
    submitMessage,
  };
};

'use client';

import { useChat } from '@ai-sdk/react';
import { useCallback, useMemo, useRef, useState } from 'react';

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
import { t } from '@/lib/i18n';
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

  // useChat captures the transport once (on the first render) and ignores any
  // transport passed on later renders, so `model` must be read at send time via
  // a ref rather than captured by value — otherwise the picked model never
  // reaches the request. The transport is memoized so it stays the one instance
  // useChat keeps.
  const modelRef = useRef(model);
  modelRef.current = model;
  const transport = useMemo(
    () => buildChatTransport(() => ({ model: modelRef.current })),
    [],
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
        // A thrown stream/transport error (no data-error part) would otherwise
        // leave the user with a blank bubble; surface a generic fallback without
        // clobbering a specific error already captured via onData. Aborts (stop)
        // never reach onError, so this will not fire on user-initiated stop.
        setActiveError(
          (prev) =>
            prev ?? { code: 'network', message: t('error.description') },
        );
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
      transport,
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
    // One Chat instance is shared across conversations, so an in-flight stream
    // must be torn down before switching away or it would append into / persist
    // against the wrong conversation.
    void stop();
    setActiveId(null);
    setMessages([]);
    setActiveError(undefined);
    convoIdRef.current = null;
  }, [setActiveId, setMessages, stop]);

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
      void stop();
      setActiveId(id);
    },
    [setActiveId, stop],
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

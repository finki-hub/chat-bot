'use client';

import { useChat } from '@ai-sdk/react';
import { posthog } from 'posthog-js';
import { useCallback, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';

import type {
  ErrorNotice,
  FeedbackType,
  MyUIMessage,
  StatusPart,
} from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { generateChatTitle } from '@/lib/chat-title';
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
  loadMessages,
  type MessageRow,
  renameConversation,
  renameConversationIfTitle,
  replaceConversationMessages,
  saveMessages,
  setMessageFeedback,
} from '@/lib/db';
import { t } from '@/lib/i18n';
import { deriveTitle } from '@/lib/messages';
import { buildChatTransport, stopChatStream } from '@/lib/transport';
import { useUiStore } from '@/lib/ui-store';
import { useConversationHydration } from '@/lib/use-conversation-hydration';
import { useConversationList } from '@/lib/use-conversation-list';
import { useStreamTiming } from '@/lib/use-stream-timing';

const fromRow = (row: MessageRow): MyUIMessage => ({
  id: row.id,
  metadata: row.metadata,
  parts: row.parts,
  role: row.role,
});

export const useConversations = (
  model: string,
  disabled = false,
  reasoning = false,
) => {
  const activeId = useUiStore((s) => s.activeConversationId);
  const setActiveId = useUiStore((s) => s.setActiveConversationId);

  const { conversations, refreshConversations } = useConversationList();
  const [activeStatus, setActiveStatus] = useState<StatusPart | undefined>();
  const [activeError, setActiveError] = useState<ErrorNotice | undefined>();
  const convoIdRef = useRef<null | string>(activeId);
  const startedAtRef = useRef<null | number>(null);
  const firstTokenAtRef = useRef<null | number>(null);
  const activeErrorRef = useRef<ErrorNotice | undefined>(undefined);
  const regeneratingMessageIdRef = useRef<null | string>(null);
  const [regeneratingMessageId, setRegeneratingMessageId] = useState<
    null | string
  >(null);
  const [generatingTitleId, setGeneratingTitleId] = useState<null | string>(
    null,
  );

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

  // useChat keeps only the first transport, so read the latest values at send time.
  const modelRef = useRef(model);
  modelRef.current = model;
  const reasoningRef = useRef(reasoning);
  reasoningRef.current = reasoning;
  const transport = useMemo(
    () =>
      buildChatTransport(() => ({
        model: modelRef.current,
        reasoning: reasoningRef.current,
      })),
    [],
  );

  const { messages, regenerate, sendMessage, setMessages, status, stop } =
    useChat<MyUIMessage>({
      id: activeId ?? undefined,
      onData: (part) => {
        switch (part.type) {
          case 'data-error':
            setActiveError(part.data);
            // Stamp it onto the message in onFinish so the notice persists past refresh.
            activeErrorRef.current = part.data;
            break;

          case 'data-reset':
            setActiveStatus(undefined);
            break;

          case 'data-status':
            setActiveStatus(part.data);
            break;
        }
      },
      onError: () => {
        regeneratingMessageIdRef.current = null;
        setRegeneratingMessageId(null);
        setActiveError(
          (prev) =>
            prev ?? { code: 'network', message: t('error.description') },
        );
      },
      onFinish: ({ isAbort, isError, message }) => {
        setActiveStatus(undefined);
        const replacementId = regeneratingMessageIdRef.current;
        // Consume on every finish so a prior turn's error can't leak onto the next.
        const error = activeErrorRef.current;
        activeErrorRef.current = undefined;
        // ai v7 also calls onFinish on abort/error; never finalize or persist a
        // partial answer (it would ghost into a switched-away conversation).
        if (isAbort || isError) {
          regeneratingMessageIdRef.current = null;
          setRegeneratingMessageId(null);
          return;
        }
        if (replacementId !== null && message.id === replacementId) {
          return;
        }
        regeneratingMessageIdRef.current = null;
        setRegeneratingMessageId(null);
        const finalizedBase = finalizeMessage(
          message,
          startedAtRef.current,
          firstTokenAtRef.current,
        );
        const withError =
          error === undefined
            ? finalizedBase
            : {
                ...finalizedBase,
                metadata: { ...finalizedBase.metadata, error },
              };
        const finalized =
          replacementId === null
            ? withError
            : { ...withError, id: replacementId };
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
      resume: activeId !== null,
      transport,
    });
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;

  useStreamTiming({
    firstTokenAtRef,
    messages,
    startedAtRef,
    status,
  });

  useConversationHydration({
    activeId,
    convoIdRef,
    model,
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  });

  const applyGeneratedTitle = useCallback(
    async (
      id: string,
      titleMessages: readonly MyUIMessage[],
      expectedTitle: string,
    ): Promise<void> => {
      setGeneratingTitleId(id);
      try {
        const title = await generateChatTitle({
          messages: titleMessages,
          queryTransformModel: modelRef.current,
        });

        if (title === null) {
          return;
        }

        await renameConversationIfTitle(id, expectedTitle, title);
        await refreshConversations();
      } finally {
        setGeneratingTitleId((current) => (current === id ? null : current));
      }
    },
    [refreshConversations],
  );

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
    [applyGeneratedTitle, model, refreshConversations, setActiveId],
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
    setActiveError(undefined);
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

  const handleGenerateTitle = useCallback(
    (id: string) => {
      const expectedTitle = conversations.find((c) => c.id === id)?.title;
      if (expectedTitle === undefined) {
        return;
      }

      const run = async (): Promise<void> => {
        const rows = await loadMessages(id);
        if (rows.length === 0) {
          return;
        }
        await applyGeneratedTitle(id, rows.map(fromRow), expectedTitle);
      };

      fireAndForget(run());
    },
    [applyGeneratedTitle, conversations],
  );

  const recordFeedback = useCallback(
    (messageId: string, vote: FeedbackType) => {
      setMessages(applyFeedback(messageId, vote));
      fireAndForget(setMessageFeedback(messageId, vote));
    },
    [setMessages],
  );

  const handleStop = useCallback(() => {
    /* eslint-disable camelcase -- PostHog event properties are snake_case. */
    posthog.capture('chat_stopped', {
      inference_model: model,
    });
    /* eslint-enable camelcase -- end of PostHog snake_case properties. */
    const stopCurrent = async (): Promise<void> => {
      const cid = convoIdRef.current;
      if (cid === null) {
        await stop();
        return;
      }
      try {
        await stopChatStream(cid);
      } finally {
        await stop();
      }
    };

    fireAndForget(stopCurrent());
  }, [model, stop]);

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
    onStop: handleStop,
    renderActions,
    retry,
    status,
    submitMessage,
  };
};

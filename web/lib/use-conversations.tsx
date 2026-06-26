'use client';

import { useChat } from '@ai-sdk/react';
import {
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';
import { fireAndForget } from '@/lib/async';
import {
  clearAllConversations,
  type ConversationRow,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  loadMessages,
  type MessageRow,
  renameConversation,
  saveMessages,
} from '@/lib/db';
import { hasText, joinText } from '@/lib/message-parts';
import { deriveTitle } from '@/lib/messages';
import { buildChatTransport } from '@/lib/transport';
import { useUiStore } from '@/lib/ui-store';

const fromRow = (row: MessageRow): MyUIMessage => ({
  id: row.id,
  metadata: row.metadata,
  parts: row.parts,
  role: row.role,
});

const priorUserText = (
  messages: MyUIMessage[],
  message: MyUIMessage,
): string | undefined => {
  const prior = messages
    .slice(0, messages.indexOf(message))
    .findLast((m) => m.role === 'user');
  return prior ? joinText(prior) : undefined;
};

const hasAssistantText = (message: MyUIMessage): boolean =>
  message.role === 'assistant' && hasText(message);

const finalizeMessage = (
  message: MyUIMessage,
  startedAt: null | number,
  firstTokenAt: null | number,
): MyUIMessage =>
  startedAt === null
    ? message
    : {
        ...message,
        metadata: {
          ...message.metadata,
          timing: {
            totalMs: Date.now() - startedAt,
            ttftMs: firstTokenAt === null ? null : firstTokenAt - startedAt,
          },
        },
      };

const renderAnswerActions =
  (
    messages: MyUIMessage[],
    status: 'error' | 'ready' | 'streaming' | 'submitted',
    regenerate: (options: { messageId: string }) => void,
  ) =>
  (message: MyUIMessage): ReactNode =>
    message.role === 'assistant' && status !== 'streaming' ? (
      <AnswerActions
        message={message}
        onRegenerate={() => {
          regenerate({ messageId: message.id });
        }}
        questionText={priorUserText(messages, message)}
      />
    ) : null;

const useStreamTiming = ({
  firstTokenAtRef,
  messages,
  setStreamStartedAt,
  startedAtRef,
  status,
}: {
  firstTokenAtRef: RefObject<null | number>;
  messages: MyUIMessage[];
  setStreamStartedAt: (value: null | number) => void;
  startedAtRef: RefObject<null | number>;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
}): void => {
  useEffect(() => {
    if (status === 'submitted') {
      const now = Date.now();
      startedAtRef.current = now;
      firstTokenAtRef.current = null;
      setStreamStartedAt(now);
    } else if (status === 'ready' || status === 'error') {
      setStreamStartedAt(null);
    }
  }, [status, firstTokenAtRef, setStreamStartedAt, startedAtRef]);

  useEffect(() => {
    if (firstTokenAtRef.current !== null || startedAtRef.current === null) {
      return;
    }
    const last = messages.at(-1);
    if (last !== undefined && hasAssistantText(last)) {
      firstTokenAtRef.current = Date.now();
    }
  }, [messages, firstTokenAtRef, startedAtRef]);
};

export const useConversations = (model: string) => {
  const activeId = useUiStore((s) => s.activeConversationId);
  const setActiveId = useUiStore((s) => s.setActiveConversationId);

  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [activeStatus, setActiveStatus] = useState<
    undefined | { label: string; tool?: string }
  >();
  const [activeError, setActiveError] = useState<
    undefined | { code: string; message: string }
  >();
  const convoIdRef = useRef<null | string>(activeId);
  const startedAtRef = useRef<null | number>(null);
  const firstTokenAtRef = useRef<null | number>(null);
  const [streamStartedAt, setStreamStartedAt] = useState<null | number>(null);

  const refreshConversations = useCallback(async () => {
    setConversations(await listConversations());
  }, []);

  const persistFinished = useCallback(
    async (message: MyUIMessage): Promise<void> => {
      const cid = convoIdRef.current;
      if (!cid) {
        return;
      }
      await saveMessages(cid, [message]);
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
      onFinish: ({ message }) => {
        setActiveStatus(undefined);
        const finalized = finalizeMessage(
          message,
          startedAtRef.current,
          firstTokenAtRef.current,
        );
        setMessages((prev) =>
          prev.map((m) => (m.id === finalized.id ? finalized : m)),
        );
        fireAndForget(persistFinished(finalized));
      },
      transport: buildChatTransport(() => ({ model })),
    });

  useEffect(() => {
    fireAndForget(refreshConversations());
  }, [refreshConversations]);

  useStreamTiming({
    firstTokenAtRef,
    messages,
    setStreamStartedAt,
    startedAtRef,
    status,
  });

  useEffect(() => {
    convoIdRef.current = activeId;
    setActiveError(undefined);
    setActiveStatus(undefined);
    let cancelled = false;
    const isCancelled = (): boolean => cancelled;
    if (activeId) {
      const hydrate = async (): Promise<void> => {
        const convo = await getConversation(activeId);
        if (!isCancelled()) {
          if (convo === undefined) {
            setActiveId(null);
          } else {
            const loaded = await loadMessages(activeId);
            if (!isCancelled()) {
              setMessages(loaded.map(fromRow));
            }
          }
        }
      };
      fireAndForget(hydrate());
    } else {
      setMessages([]);
    }

    return () => {
      cancelled = true;
    };
  }, [activeId, setActiveId, setMessages]);

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
    fireAndForget(regenerate());
  }, [regenerate]);

  const handleRename = useCallback(
    async (id: string, title: string) => {
      await renameConversation(id, title);
      await refreshConversations();
    },
    [refreshConversations],
  );

  return {
    activeError,
    activeId,
    activeStatus,
    conversations,
    messages,
    onClearAll: handleClearAll,
    onDelete: handleDelete,
    onNewChat: handleNewChat,
    onRename: handleRename,
    onSelect: handleSelect,
    onStop: stop,
    renderActions: renderAnswerActions(messages, status, regenerate),
    retry,
    status,
    streamStartedAt,
    submitMessage,
  };
};

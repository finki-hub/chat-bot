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
  type ConversationRow,
  createConversation,
  deleteConversation,
  listConversations,
  loadMessages,
  type MessageRow,
  renameConversation,
  saveMessages,
} from '@/lib/db';
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
): string | undefined =>
  messages
    .slice(0, messages.indexOf(message))
    .findLast((m) => m.role === 'user')
    ?.parts.filter(
      (p): p is { text: string; type: 'text' } => p.type === 'text',
    )
    .map((p) => p.text)
    .join('');

const hasAssistantText = (message: MyUIMessage): boolean =>
  message.role === 'assistant' &&
  message.parts.some(
    (p): p is { text: string; type: 'text' } =>
      p.type === 'text' && p.text.length > 0,
  );

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
        const startedAt = startedAtRef.current;
        const finalized: MyUIMessage =
          startedAt === null
            ? message
            : {
                ...message,
                metadata: {
                  ...message.metadata,
                  timing: {
                    totalMs: Date.now() - startedAt,
                    ttftMs:
                      firstTokenAtRef.current === null
                        ? null
                        : firstTokenAtRef.current - startedAt,
                  },
                },
              };
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
    if (activeId) {
      const hydrate = async (): Promise<void> => {
        const loaded = await loadMessages(activeId);
        if (!cancelled) {
          setMessages(loaded.map(fromRow));
        }
      };
      fireAndForget(hydrate());
    } else {
      setMessages([]);
    }

    return () => {
      cancelled = true;
    };
  }, [activeId, setMessages]);

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

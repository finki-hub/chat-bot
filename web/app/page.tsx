'use client';

import { useChat } from '@ai-sdk/react';
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';
import { Composer } from '@/components/chat/composer';
import { Thread } from '@/components/chat/thread';
import { Header } from '@/components/shell/header';
import { Sidebar } from '@/components/shell/sidebar';
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
import { useModels } from '@/lib/use-models';

const fromRow = (row: MessageRow): MyUIMessage => ({
  id: row.id,
  metadata: row.metadata,
  parts: row.parts,
  role: row.role,
});

// The user turn that precedes a given assistant message, flattened to text.
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

// Per-answer actions, shown only for finished assistant turns (spec §9).
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

const ChatScreen = () => {
  const activeId = useUiStore((s) => s.activeConversationId);
  const setActiveId = useUiStore((s) => s.setActiveConversationId);
  const model = useUiStore((s) => s.model);
  const setModel = useUiStore((s) => s.setModel);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  const { data: modelList } = useModels();
  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [activeStatus, setActiveStatus] = useState<
    undefined | { label: string; tool?: string }
  >();
  const [activeError, setActiveError] = useState<
    undefined | { code: string; message: string }
  >();
  const convoIdRef = useRef<null | string>(activeId);

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
        fireAndForget(persistFinished(message));
      },
      transport: buildChatTransport(() => ({ model })),
    });

  useEffect(() => {
    fireAndForget(refreshConversations());
  }, [refreshConversations]);

  // Hydrate when the active conversation changes.
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
        // The id comes from the just-resolved create, not stale state; the ref
        // write is intentionally synchronous with it.
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

  return (
    <div className="flex h-dvh w-full flex-col">
      <Header onToggleSidebar={toggleSidebar} />
      <div className="flex min-h-0 flex-1">
        <Sidebar
          activeId={activeId}
          conversations={conversations}
          onDelete={handleDelete}
          onNewChat={handleNewChat}
          onRename={handleRename}
          onSelect={handleSelect}
          open={sidebarOpen}
        />
        <main className="flex min-w-0 flex-1 flex-col">
          <Thread
            activeError={activeError}
            activeStatus={activeStatus}
            messages={messages}
            onRetry={retry}
            renderActions={renderAnswerActions(messages, status, regenerate)}
            status={status}
          />
          <Composer
            model={model}
            models={modelList ?? []}
            onModelChange={setModel}
            onStop={stop}
            onSubmit={submitMessage}
            status={status}
          />
        </main>
      </div>
    </div>
  );
};

const ChatPage = () => <ChatScreen />;

export default ChatPage;

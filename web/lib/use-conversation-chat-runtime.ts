'use client';

import { useChat } from '@ai-sdk/react';
import { type RefObject, useEffect, useMemo, useRef, useState } from 'react';

import type { ErrorNotice, MyUIMessage, StatusPart } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import {
  finalizeMessage,
  replaceFinishedMessage,
} from '@/lib/conversation-message-state';
import { t } from '@/lib/i18n';
import { isSponsoredModel } from '@/lib/model-catalog';
import { buildChatTransport } from '@/lib/transport';
import { useConversationHydration } from '@/lib/use-conversation-hydration';
import { useModels } from '@/lib/use-models';
import { useStreamTiming } from '@/lib/use-stream-timing';

type UseConversationChatRuntimeOptions = {
  readonly activeId: null | string;
  readonly model: string;
  readonly preserveEmptyHydrationIdRef: RefObject<null | string>;
  readonly reasoning: boolean;
  readonly refreshConversations: () => Promise<void>;
  readonly setActiveId: (id: null | string) => void;
};

export const useConversationChatRuntime = ({
  activeId,
  model,
  preserveEmptyHydrationIdRef,
  reasoning,
  refreshConversations,
  setActiveId,
}: UseConversationChatRuntimeOptions) => {
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
  const { models, refetch: refetchModels } = useModels();
  const refetchModelsRef = useRef(refetchModels);
  refetchModelsRef.current = refetchModels;
  const sponsoredModelRef = useRef(false);
  sponsoredModelRef.current = models.some(
    (entry) => entry.id === model && isSponsoredModel(entry),
  );
  const sponsoredRefreshHandledRef = useRef(false);

  const refreshSponsoredModels = () => {
    if (!sponsoredModelRef.current || sponsoredRefreshHandledRef.current) {
      return;
    }
    sponsoredRefreshHandledRef.current = true;
    fireAndForget(refetchModelsRef.current());
  };

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
        refreshSponsoredModels();
        regeneratingMessageIdRef.current = null;
        setRegeneratingMessageId(null);
        setActiveError(
          (prev) =>
            prev ?? { code: 'network', message: t('error.description') },
        );
      },
      onFinish: ({ isAbort, isError, message }) => {
        if (!isAbort) {
          refreshSponsoredModels();
        }
        const finishedConversationId = activeId;
        setActiveStatus(undefined);
        const replacementId = regeneratingMessageIdRef.current;
        const error = activeErrorRef.current;
        activeErrorRef.current = undefined;
        if (isAbort || isError) {
          regeneratingMessageIdRef.current = null;
          setRegeneratingMessageId(null);
          return;
        }
        if (finishedConversationId === null) {
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
          fireAndForget(refreshConversations());
          return next;
        });
      },
      resume: activeId !== null,
      transport,
    });
  useEffect(() => {
    if (status === 'submitted' || status === 'streaming') {
      sponsoredRefreshHandledRef.current = false;
    }
  }, [status]);
  const activeStreamConversationIdRef = useRef<null | string>(null);
  if (activeId !== null && status !== 'ready') {
    activeStreamConversationIdRef.current = activeId;
  }
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;

  useStreamTiming({
    firstTokenAtRef,
    messages,
    startedAtRef,
    status,
  });

  const hydratingConversation = useConversationHydration({
    activeId,
    activeStreamConversationIdRef,
    convoIdRef,
    preserveEmptyHydrationIdRef,
    setActiveError,
    setActiveId,
    setActiveStatus,
    setMessages,
  });

  return {
    activeError,
    activeStatus,
    convoIdRef,
    hydratingConversation,
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
  };
};

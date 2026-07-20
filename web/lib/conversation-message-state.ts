import type { FeedbackSelection, MyUIMessage } from '@/lib/api-types';
import type { ActiveConversationStream } from '@/lib/conversation-types';

type FinishedReplacement = {
  readonly pruneAfterReplacement: boolean;
  readonly replacement: MyUIMessage;
  readonly streamMessageId: string;
};

type HydrationReconciliation = {
  readonly activeStream: ActiveConversationStream | null;
  readonly current: readonly MyUIMessage[];
  readonly persisted: readonly MyUIMessage[];
};

export const finalizeMessage = (
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

export const replaceFinishedMessage =
  ({
    pruneAfterReplacement,
    replacement,
    streamMessageId,
  }: FinishedReplacement): ((messages: MyUIMessage[]) => MyUIMessage[]) =>
  (messages) => {
    const responseId = replacement.metadata?.responseId;
    const canDedupeByResponseId = replacement.id === streamMessageId;
    const replacementIndex = messages.findIndex((message) => {
      if (message.id === replacement.id) {
        return true;
      }

      return (
        canDedupeByResponseId &&
        responseId !== undefined &&
        message.role === 'assistant' &&
        message.id !== streamMessageId &&
        message.metadata?.responseId === responseId
      );
    });

    if (replacementIndex === -1) {
      return [
        ...messages.filter((message) => message.id !== streamMessageId),
        replacement,
      ];
    }

    const existingReplacement = messages.at(replacementIndex);
    const normalizedReplacement =
      existingReplacement === undefined
        ? replacement
        : {
            ...replacement,
            id: existingReplacement.id,
            metadata: {
              ...existingReplacement.metadata,
              ...replacement.metadata,
            },
          };

    return messages.flatMap((message, messageIndex): MyUIMessage[] => {
      if (messageIndex > replacementIndex && pruneAfterReplacement) {
        return [];
      }

      if (messageIndex === replacementIndex) {
        return [normalizedReplacement];
      }

      if (message.id === streamMessageId) {
        return [];
      }

      return [message];
    });
  };

export const previewRegeneration = (
  messages: MyUIMessage[],
  messageId: null | string,
): MyUIMessage[] => {
  if (messageId === null) {
    return messages;
  }

  const targetIndex = messages.findIndex((message) => message.id === messageId);
  if (targetIndex === -1) {
    return messages;
  }

  const target = messages.at(targetIndex);

  if (target === undefined) {
    return messages;
  }

  return [
    ...messages.slice(0, targetIndex),
    { ...target, metadata: {}, parts: [] },
  ];
};

export const reconcileHydratedMessages = ({
  activeStream,
  current,
  persisted,
}: HydrationReconciliation): MyUIMessage[] => {
  if (activeStream === null) {
    return [...persisted];
  }
  const resumedAssistant = current.find(
    (message) =>
      message.role === 'assistant' &&
      message.metadata?.responseId === activeStream.id,
  );
  if (resumedAssistant === undefined) {
    return previewRegeneration(
      [...persisted],
      activeStream.replacementMessageId,
    );
  }
  const replacement =
    activeStream.replacementMessageId === null
      ? resumedAssistant
      : { ...resumedAssistant, id: activeStream.replacementMessageId };
  return replaceFinishedMessage({
    pruneAfterReplacement: activeStream.replacementMessageId !== null,
    replacement,
    streamMessageId: resumedAssistant.id,
  })([...persisted]);
};

export const applyFeedback =
  (messageId: string, feedback: FeedbackSelection) =>
  (messages: MyUIMessage[]): MyUIMessage[] =>
    messages.map((message) => {
      if (message.id !== messageId) {
        return message;
      }
      if (feedback !== null) {
        return { ...message, metadata: { ...message.metadata, feedback } };
      }
      const metadata = { ...message.metadata };
      Reflect.deleteProperty(metadata, 'feedback');
      return { ...message, metadata };
    });

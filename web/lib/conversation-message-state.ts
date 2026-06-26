import type { MyUIMessage } from '@/lib/api-types';

type FinishedReplacement = {
  readonly pruneAfterReplacement: boolean;
  readonly replacement: MyUIMessage;
  readonly streamMessageId: string;
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
    const replacementIndex = messages.findIndex(
      (message) => message.id === replacement.id,
    );

    if (replacementIndex === -1) {
      return [
        ...messages.filter((message) => message.id !== streamMessageId),
        replacement,
      ];
    }

    return messages.flatMap((message, messageIndex): MyUIMessage[] => {
      if (messageIndex > replacementIndex && pruneAfterReplacement) {
        return [];
      }

      if (message.id === replacement.id) {
        return [replacement];
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

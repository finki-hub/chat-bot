import {
  MAX_CHARS_PER_TURN,
  MAX_MESSAGES,
  type MyUIMessage,
} from '@/lib/api-types';

const FALLBACK_TITLE = 'Нов разговор';
const TITLE_MAX = 60;

const textLength = (message: MyUIMessage): number => {
  let total = 0;

  for (const part of message.parts) {
    if (part.type === 'text') {
      total += part.text.length;
    }
  }

  return total;
};

const capTextParts = (message: MyUIMessage): MyUIMessage => {
  if (textLength(message) <= MAX_CHARS_PER_TURN) {
    return message;
  }

  let budget = MAX_CHARS_PER_TURN;
  const parts = message.parts.map((part) => {
    if (part.type !== 'text') {
      return part;
    }

    if (budget <= 0) {
      return { ...part, text: '' };
    }

    const slice = part.text.slice(0, budget);

    budget -= slice.length;

    return { ...part, text: slice };
  });

  return { ...message, parts };
};

export const trimForRequest = (messages: MyUIMessage[]): MyUIMessage[] => {
  const windowed =
    messages.length > MAX_MESSAGES
      ? messages.slice(messages.length - MAX_MESSAGES)
      : messages;

  return windowed.map(capTextParts);
};

export const deriveTitle = (firstUserText: string): string => {
  const firstLine = firstUserText.split('\n', 1)[0]?.trim() ?? '';

  if (firstLine.length === 0) {
    return FALLBACK_TITLE;
  }

  if (firstLine.length <= TITLE_MAX) {
    return firstLine;
  }

  return `${firstLine.slice(0, TITLE_MAX - 1)}…`;
};

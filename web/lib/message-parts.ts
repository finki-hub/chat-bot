import type { MyUIMessage } from '@/lib/api-types';

type MessagePart = MyUIMessage['parts'][number];
type ReasoningPart = {
  state?: 'done' | 'streaming';
  text: string;
  type: 'reasoning';
};
type TextPart = { text: string; type: 'text' };

const isTextPart = (part: MessagePart): part is TextPart =>
  part.type === 'text';

const isReasoningPart = (part: MessagePart): part is ReasoningPart =>
  part.type === 'reasoning';

export const textParts = (message: MyUIMessage): TextPart[] =>
  message.parts.filter(isTextPart);

export const reasoningParts = (message: MyUIMessage): ReasoningPart[] =>
  message.parts.filter(isReasoningPart);

export const joinText = (message: MyUIMessage): string =>
  textParts(message)
    .map((part) => part.text)
    .join('');

export const lastText = (message: MyUIMessage): null | string =>
  textParts(message).at(-1)?.text ?? null;

export const hasText = (message: MyUIMessage): boolean =>
  textParts(message).some((part) => part.text.length > 0);

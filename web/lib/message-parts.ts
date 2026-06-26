import type { MyUIMessage } from '@/lib/api-types';

type MessagePart = MyUIMessage['parts'][number];
type TextPart = { text: string; type: 'text' };

const isTextPart = (part: MessagePart): part is TextPart =>
  part.type === 'text';

export const textParts = (message: MyUIMessage): TextPart[] =>
  message.parts.filter(isTextPart);

export const joinText = (message: MyUIMessage): string =>
  textParts(message)
    .map((part) => part.text)
    .join('');

export const lastText = (message: MyUIMessage): null | string =>
  textParts(message).at(-1)?.text ?? null;

export const hasText = (message: MyUIMessage): boolean =>
  textParts(message).some((part) => part.text.length > 0);

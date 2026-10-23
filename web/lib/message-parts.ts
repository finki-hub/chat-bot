import type { DynamicToolUIPart } from 'ai';

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

export const isToolCallPart = (part: MessagePart): part is DynamicToolUIPart =>
  part.type === 'dynamic-tool';

const isReasoningPart = (part: MessagePart): part is ReasoningPart =>
  part.type === 'reasoning';

export const textParts = (message: MyUIMessage): TextPart[] =>
  message.parts.filter(isTextPart);

export const reasoningParts = (message: MyUIMessage): ReasoningPart[] =>
  message.parts.filter(isReasoningPart);

export const toolCallParts = (message: MyUIMessage): DynamicToolUIPart[] =>
  message.parts.filter(isToolCallPart);

export const hasToolCalls = (message: MyUIMessage): boolean =>
  message.parts.some(isToolCallPart);

// Text around tool calls forms separate segments of one answer; without tool calls,
// extra text parts come from legacy streams and are joined as before.
export const joinText = (message: MyUIMessage): string =>
  textParts(message)
    .map((part) => part.text)
    .filter((text) => !hasToolCalls(message) || text.trim() !== '')
    .join(hasToolCalls(message) ? '\n\n' : '');

export const lastText = (message: MyUIMessage): null | string =>
  textParts(message).at(-1)?.text ?? null;

// The answer a user copies or a title is generated from: every segment when the
// message has tool calls, otherwise the last text part (earlier ones are a legacy
// pre-tool preamble that the UI hides).
export const answerText = (message: MyUIMessage): null | string =>
  hasToolCalls(message) ? joinText(message) || null : lastText(message);

export const hasText = (message: MyUIMessage): boolean =>
  textParts(message).some((part) => part.text.length > 0);

export const hasReasoning = (message: MyUIMessage): boolean =>
  reasoningParts(message).some((part) => part.text.length > 0);

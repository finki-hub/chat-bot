import type { DynamicToolUIPart } from 'ai';

import type { MyUIMessage } from '@/lib/api-types';

import { t } from '@/lib/i18n';
import { isToolCallPart } from '@/lib/message-parts';

export type ToolCallStatus = 'done' | 'error' | 'incomplete' | 'running';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isResolved = (part: DynamicToolUIPart): boolean =>
  part.state === 'output-available' || part.state === 'output-error';

// Unresolved calls the model has already answered past: answer text follows them, so
// they will not resolve any more. A later call resolving first proves nothing, since
// parallel calls finish in any order.
export const supersededToolCalls = (message: MyUIMessage): Set<string> => {
  const superseded = new Set<string>();
  let answered = false;

  for (const part of message.parts.toReversed()) {
    if (part.type === 'text' && part.text.trim() !== '') {
      answered = true;
    } else if (answered && isToolCallPart(part) && !isResolved(part)) {
      superseded.add(part.toolCallId);
    }
  }

  return superseded;
};

// A call still waiting for its result after the stream ended or moved on was interrupted.
export const toolCallStatus = (
  part: DynamicToolUIPart,
  running: boolean,
): ToolCallStatus => {
  if (part.state === 'output-available') {
    return 'done';
  }

  if (part.state === 'output-error') {
    return 'error';
  }

  return running ? 'running' : 'incomplete';
};

export const toolCallInputText = (input: unknown): null | string => {
  if (!isRecord(input) || Object.keys(input).length === 0) {
    return null;
  }

  const { query } = input;

  if (typeof query !== 'string') {
    return JSON.stringify(input);
  }

  return query.trim() === '' ? null : query;
};

export const toolCallResultText = (part: DynamicToolUIPart): null | string => {
  if (part.state === 'output-error') {
    return part.errorText.length > 0 ? part.errorText : t('toolCall.failed');
  }

  if (part.state !== 'output-available' || !isRecord(part.output)) {
    return null;
  }

  const { summary } = part.output;

  return typeof summary === 'string' && summary.length > 0 ? summary : null;
};

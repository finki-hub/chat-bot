import type { DynamicToolUIPart } from 'ai';

import { t } from '@/lib/i18n';

export type ToolCallStatus = 'done' | 'error' | 'incomplete' | 'running';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// A call still waiting for its result after the stream ended was interrupted.
export const toolCallStatus = (
  part: DynamicToolUIPart,
  pending: boolean,
): ToolCallStatus => {
  if (part.state === 'output-available') {
    return 'done';
  }

  if (part.state === 'output-error') {
    return 'error';
  }

  return pending ? 'running' : 'incomplete';
};

export const toolCallInputText = (input: unknown): null | string => {
  if (!isRecord(input) || Object.keys(input).length === 0) {
    return null;
  }

  const { query } = input;

  return typeof query === 'string' ? query : JSON.stringify(input);
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

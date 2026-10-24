import type { DynamicToolUIPart } from 'ai';

import type { MyUIMessage } from '@/lib/api-types';

import { t } from '@/lib/i18n';
import { isToolCallPart } from '@/lib/message-parts';
import { isSearchTool } from '@/lib/search-status';

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

export const countLabel = (count: number, one: string, many: string): string =>
  `${count} ${count === 1 ? one : many}`;

const SOURCE_COUNT_RE = /\d+/u;

// Compact result for a search row header, read from the API summaries
// ("Пронајдени извори: 6", "Нема пронајдени извори", "...делумно поврзани извори").
export const toolCallResultBadge = (part: DynamicToolUIPart): null | string => {
  if (part.state !== 'output-available' || !isSearchTool(part.toolName)) {
    return null;
  }

  const summary = toolCallResultText(part);

  if (summary === null) {
    return null;
  }

  const count = SOURCE_COUNT_RE.exec(summary)?.[0];

  if (count !== undefined) {
    return countLabel(
      Number(count),
      t('toolCall.sourceOne'),
      t('toolCall.sourceMany'),
    );
  }

  if (summary.startsWith('Нема')) {
    return t('toolCall.noSources');
  }

  return summary.includes('делумно') ? t('toolCall.partialSources') : null;
};

export type ToolCallTally = {
  problems: number;
  searches: number;
  tools: number;
};

// Counts for a finished message, where every unresolved call was interrupted.
export const tallyToolCalls = (
  parts: readonly DynamicToolUIPart[],
): ToolCallTally => {
  const tally: ToolCallTally = { problems: 0, searches: 0, tools: 0 };

  for (const part of parts) {
    if (isSearchTool(part.toolName)) {
      tally.searches += 1;
    } else {
      tally.tools += 1;
    }

    if (toolCallStatus(part, false) !== 'done') {
      tally.problems += 1;
    }
  }

  return tally;
};

import {
  type ChatRequestBody,
  type ConversationTurn,
  type ErrorNotice,
  MAX_CHARS_PER_TURN,
  type MessageDiagnostics,
  type MyUIMessage,
  type RetrievedSource,
} from '@/lib/api-types';
import { joinText } from '@/lib/message-parts';
import { type ParsedEvent } from '@/lib/sse';

export type ChatClientBody = {
  embeddingsModel?: string;
  maxTokens?: number;
  messageId?: string;
  messages: MyUIMessage[];
  model?: string;
  posthogDistinctId?: string;
  posthogSessionId?: string;
  reasoning?: boolean;
  temperature?: number;
  topP?: number;
  trigger?: string;
};

export type ToolCallOutput = { summary: string };

export type UiStreamMeta = {
  inferenceModel?: string;
  replacementMessageId?: string;
  responseId?: string;
};

export type UiStreamPart =
  | {
      data: ErrorNotice;
      transient: true;
      type: 'data-error';
    }
  | {
      data: Record<string, never>;
      transient: true;
      type: 'data-reset';
    }
  | {
      data: { label: string; stage?: string; tool?: string };
      transient: true;
      type: 'data-status';
    }
  | { delta: string; id: string; type: 'reasoning-delta' }
  | { delta: string; id: string; type: 'text-delta' }
  | {
      dynamic: true;
      errorText: string;
      toolCallId: string;
      type: 'tool-output-error';
    }
  | {
      dynamic: true;
      input: Record<string, unknown>;
      title: string;
      toolCallId: string;
      toolName: string;
      type: 'tool-input-available';
    }
  | {
      dynamic: true;
      output: ToolCallOutput;
      toolCallId: string;
      type: 'tool-output-available';
    }
  | { id: string; type: 'reasoning-end' }
  | { id: string; type: 'reasoning-start' }
  | { id: string; type: 'text-end' }
  | { id: string; type: 'text-start' }
  | { messageId?: string; messageMetadata: UiStreamMeta; type: 'start' }
  | {
      messageMetadata: {
        diagnostics?: MessageDiagnostics;
        sources?: readonly RetrievedSource[];
      };
      type: 'message-metadata';
    };

export type UiStreamWriter = {
  write: (part: UiStreamPart) => void;
};

const messagesForRequest = (body: ChatClientBody): readonly MyUIMessage[] => {
  if (body.messageId === undefined || !body.trigger?.includes('regenerate')) {
    return body.messages;
  }

  const messageIndex = body.messages.findIndex(
    (message) => message.id === body.messageId,
  );

  return messageIndex === -1
    ? body.messages
    : body.messages.slice(0, messageIndex);
};

export const currentUserMessageForRequest = (
  body: ChatClientBody,
): MyUIMessage | undefined =>
  messagesForRequest(body).findLast((message) => message.role === 'user');

export const toChatRequestBody = (body: ChatClientBody): ChatRequestBody => {
  const userMessage = currentUserMessageForRequest(body);
  const trimmed = userMessage === undefined ? [] : [userMessage];
  const messages: ConversationTurn[] = trimmed.map((message) => {
    const content = joinText(message);

    return { content: content.slice(0, MAX_CHARS_PER_TURN), role: 'user' };
  });

  return {
    interface: 'web',
    messages,
    /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
    ...(body.embeddingsModel !== undefined && {
      embeddings_model: body.embeddingsModel,
    }),
    ...(body.model !== undefined && { inference_model: body.model }),
    ...(body.maxTokens !== undefined && { max_tokens: body.maxTokens }),
    ...(body.reasoning !== undefined && { reasoning: body.reasoning }),
    ...(body.temperature !== undefined && { temperature: body.temperature }),
    ...(body.topP !== undefined && { top_p: body.topP }),
    /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */
  };
};

// One factory for the text and reasoning parts. Per-branch literal objects (not a
// templated `${kind}-start`) keep the discriminated UiStreamPart union narrowable.
const createStreamPart = (
  writer: UiStreamWriter,
  idGen: () => string,
  kind: 'reasoning' | 'text',
) => {
  let id: null | string = null;

  return {
    appendDelta(delta: string): void {
      const ensuredId = id ?? idGen();

      if (id === null) {
        id = ensuredId;
        writer.write(
          kind === 'text'
            ? { id: ensuredId, type: 'text-start' }
            : { id: ensuredId, type: 'reasoning-start' },
        );
      }

      writer.write(
        kind === 'text'
          ? { delta, id: ensuredId, type: 'text-delta' }
          : { delta, id: ensuredId, type: 'reasoning-delta' },
      );
    },
    end(): void {
      if (id === null) {
        return;
      }

      writer.write(
        kind === 'text'
          ? { id, type: 'text-end' }
          : { id, type: 'reasoning-end' },
      );
      id = null;
    },
  };
};

const createToolCalls = (writer: UiStreamWriter) => {
  type ToolResultEvent = Extract<ParsedEvent, { type: 'tool_result' }>;
  type ToolCallEvent = Extract<ParsedEvent, { type: 'tool_call' }>;
  type StatusEvent = Extract<ParsedEvent, { type: 'status' }>;
  const seen = new Set<string>();
  // Owns the tool lifecycle and the `status` frames that duplicate it. The SDK throws on
  // a result for an unknown call id, so only pending calls resolve.
  const pending = new Set<string>();
  let statusShown = false;

  return {
    call(event: ToolCallEvent): void {
      seen.add(event.id);
      pending.add(event.id);
      writer.write({
        dynamic: true,
        input: event.input,
        title: event.label,
        toolCallId: event.id,
        toolName: event.name,
        type: 'tool-input-available',
      });

      // The tool row replaces the status sent just before it; clear it so it
      // does not linger as the active status.
      if (!statusShown) {
        return;
      }

      statusShown = false;
      writer.write({ data: {}, transient: true, type: 'data-reset' });
    },
    clearStatus(): void {
      statusShown = false;
    },
    isNew: (id: string): boolean => !seen.has(id),
    result(event: ToolResultEvent): void {
      if (!pending.delete(event.id)) {
        return;
      }

      writer.write(
        event.ok
          ? {
              dynamic: true,
              output: { summary: event.summary },
              toolCallId: event.id,
              type: 'tool-output-available',
            }
          : {
              dynamic: true,
              errorText: event.summary,
              toolCallId: event.id,
              type: 'tool-output-error',
            },
      );
    },
    status(event: StatusEvent): void {
      // Tool statuses duplicate the inline tool parts once the API sends them.
      if (event.tool !== undefined && seen.size > 0) {
        return;
      }

      statusShown = true;
      writer.write({
        data: {
          label: event.label,
          ...(event.stage !== undefined && { stage: event.stage }),
          ...(event.tool !== undefined && { tool: event.tool }),
        },
        transient: true,
        type: 'data-status',
      });
    },
  };
};

const drain = async (
  events: AsyncIterable<ParsedEvent>,
  handle: (event: ParsedEvent) => void,
): Promise<void> => {
  for await (const event of events) {
    handle(event);
  }
};

/* eslint-disable @typescript-eslint/max-params -- spec-mandated signature: (events, writer, meta, idGen?); idGen is injected only in tests */
export const translateToUiStream = async (
  events: AsyncIterable<ParsedEvent>,
  writer: UiStreamWriter,
  meta: UiStreamMeta,
  idGen: () => string = () => crypto.randomUUID(),
): Promise<void> => {
  /* eslint-enable @typescript-eslint/max-params -- re-enable once past the declaration */
  writer.write({
    ...(meta.replacementMessageId !== undefined && {
      messageId: meta.replacementMessageId,
    }),
    messageMetadata: meta,
    type: 'start',
  });

  const textPart = createStreamPart(writer, idGen, 'text');
  const reasoningPart = createStreamPart(writer, idGen, 'reasoning');
  const toolCalls = createToolCalls(writer);
  let stopped = false;

  const handleEvent = (event: ParsedEvent): void => {
    switch (event.type) {
      case 'done':
        reasoningPart.end();
        textPart.end();
        break;

      case 'error':
        writer.write({
          data: {
            code: event.code,
            message: event.message,
            ...(event.resets_at !== undefined && {
              // eslint-disable-next-line camelcase -- mirrors the Python SSE wire contract.
              resets_at: event.resets_at,
            }),
          },
          transient: true,
          type: 'data-error',
        });

        if (event.code !== 'interrupted') {
          reasoningPart.end();
          textPart.end();
          stopped = true;
        }

        break;

      case 'meta':
        writer.write({
          messageMetadata: { diagnostics: event.diagnostics },
          type: 'message-metadata',
        });
        break;

      case 'reset':
        // `reset` only delimits the dropped pre-tool text preamble — never reasoning.
        textPart.end();
        toolCalls.clearStatus();
        writer.write({ data: {}, transient: true, type: 'data-reset' });
        break;

      case 'sources':
        writer.write({
          messageMetadata: { sources: event.sources },
          type: 'message-metadata',
        });
        break;

      case 'status':
        toolCalls.status(event);
        break;

      case 'thinking':
        if (!stopped) {
          reasoningPart.appendDelta(event.text);
        }

        break;

      case 'token':
        if (!stopped) {
          reasoningPart.end();
          textPart.appendDelta(event.text);
        }

        break;

      case 'tool_call':
        if (!stopped && toolCalls.isNew(event.id)) {
          // Ending the open parts makes text after the call start a new segment.
          reasoningPart.end();
          textPart.end();
          toolCalls.call(event);
        }

        break;

      case 'tool_result':
        toolCalls.result(event);
        break;

      default:
        break;
    }
  };

  await drain(events, handleEvent);

  reasoningPart.end();
  textPart.end();
};

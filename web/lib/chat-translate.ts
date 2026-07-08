import {
  type ChatRequestBody,
  type ConversationTurn,
  type ErrorNotice,
  MAX_CHARS_PER_TURN,
  type MessageDiagnostics,
  type MyUIMessage,
  type QueryTransformMode,
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
  queryTransformMode?: QueryTransformMode;
  queryTransformModel?: string;
  reasoning?: boolean;
  temperature?: number;
  topP?: number;
  trigger?: string;
};

export type UiStreamMeta = { inferenceModel?: string; responseId?: string };

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
  | { id: string; type: 'reasoning-end' }
  | { id: string; type: 'reasoning-start' }
  | { id: string; type: 'text-end' }
  | { id: string; type: 'text-start' }
  | { messageMetadata: UiStreamMeta; type: 'start' }
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
    ...(body.queryTransformModel !== undefined && {
      query_transform_model: body.queryTransformModel,
    }),
    ...(body.queryTransformMode !== undefined && {
      query_transform_mode: body.queryTransformMode,
    }),
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
  writer.write({ messageMetadata: meta, type: 'start' });

  const textPart = createStreamPart(writer, idGen, 'text');
  const reasoningPart = createStreamPart(writer, idGen, 'reasoning');
  let stopped = false;

  const handleEvent = (event: ParsedEvent): void => {
    switch (event.type) {
      case 'done':
        reasoningPart.end();
        textPart.end();
        break;

      case 'error':
        writer.write({
          data: { code: event.code, message: event.message },
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
        writer.write({ data: {}, transient: true, type: 'data-reset' });
        break;

      case 'sources':
        writer.write({
          messageMetadata: { sources: event.sources },
          type: 'message-metadata',
        });
        break;

      case 'status':
        writer.write({
          data: {
            label: event.label,
            ...(event.stage !== undefined && { stage: event.stage }),
            ...(event.tool !== undefined && { tool: event.tool }),
          },
          transient: true,
          type: 'data-status',
        });
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

      default:
        break;
    }
  };

  await drain(events, handleEvent);

  reasoningPart.end();
  textPart.end();
};

import {
  type ChatRequestBody,
  type ConversationTurn,
  MAX_CHARS_PER_TURN,
  MAX_MESSAGES,
  type MyUIMessage,
} from '@/lib/api-types';
import { type ParsedEvent } from '@/lib/sse';

export type ChatClientBody = {
  embeddingsModel?: string;
  maxTokens?: number;
  messages: MyUIMessage[];
  model?: string;
  queryTransformModel?: string;
  temperature?: number;
  topP?: number;
  userId?: string;
};

export type UiStreamMeta = { inferenceModel?: string; responseId?: string };

export type UiStreamPart =
  | {
      data: { code: string; message: string };
      transient: true;
      type: 'data-error';
    }
  | {
      data: { label: string; tool?: string };
      transient: true;
      type: 'data-status';
    }
  | { delta: string; id: string; type: 'text-delta' }
  | { id: string; type: 'text-end' }
  | { id: string; type: 'text-start' }
  | { messageMetadata: UiStreamMeta; type: 'start' };

export type UiStreamWriter = {
  write: (part: UiStreamPart) => void;
};

const joinText = (message: MyUIMessage): string => {
  const text = message.parts
    .filter(
      (part): part is { text: string; type: 'text' } => part.type === 'text',
    )
    .map((part) => part.text)
    .join('');

  return text.length > MAX_CHARS_PER_TURN
    ? text.slice(0, MAX_CHARS_PER_TURN)
    : text;
};

export const toChatRequestBody = (body: ChatClientBody): ChatRequestBody => {
  const trimmed = body.messages.slice(-MAX_MESSAGES);
  const messages: ConversationTurn[] = trimmed.map((message) => ({
    content: joinText(message),
    role: message.role === 'assistant' ? 'assistant' : 'user',
  }));

  return {
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
    ...(body.temperature !== undefined && { temperature: body.temperature }),
    ...(body.topP !== undefined && { top_p: body.topP }),
    /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */
  };
};

const createTextPart = (writer: UiStreamWriter, idGen: () => string) => {
  let id: null | string = null;

  return {
    appendDelta(delta: string): void {
      const ensuredId = id ?? idGen();

      if (id === null) {
        id = ensuredId;
        writer.write({ id: ensuredId, type: 'text-start' });
      }

      writer.write({ delta, id: ensuredId, type: 'text-delta' });
    },
    end(): void {
      if (id === null) {
        return;
      }

      writer.write({ id, type: 'text-end' });
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

  const textPart = createTextPart(writer, idGen);
  let stopped = false;

  const handleEvent = (event: ParsedEvent): void => {
    switch (event.type) {
      case 'done':
        textPart.end();
        break;

      case 'error':
        writer.write({
          data: { code: event.code, message: event.message },
          transient: true,
          type: 'data-error',
        });

        if (event.code !== 'interrupted') {
          textPart.end();
          stopped = true;
        }

        break;

      case 'reset':
        textPart.end();
        break;

      case 'status':
        writer.write({
          data: {
            label: event.label,
            ...(event.tool !== undefined && { tool: event.tool }),
          },
          transient: true,
          type: 'data-status',
        });
        break;

      case 'token':
        if (!stopped) {
          textPart.appendDelta(event.text);
        }

        break;

      default:
        break;
    }
  };

  await drain(events, handleEvent);

  textPart.end();
};

import type { ChatErrorCode } from '@/lib/api-types';

export type ParsedEvent =
  | { code: ChatErrorCode; message: string; type: 'error' }
  | { label: string; state: string; tool?: string; type: 'status' }
  | { text: string; type: 'token' }
  | { type: 'done' }
  | { type: 'reset' };

export type SseSource =
  | AsyncIterable<string | Uint8Array>
  | ReadableStream<Uint8Array>;

const ERROR_CODES: ReadonlySet<ChatErrorCode> = new Set([
  'agent_error',
  'interrupted',
  'no_answer',
]);

const TRAILING_CR = /\r$/u;
const LEADING_SPACE = /^ /u;

const toErrorCode = (value: unknown): ChatErrorCode =>
  ERROR_CODES.has(value as ChatErrorCode)
    ? (value as ChatErrorCode)
    : 'agent_error';

const asString = (value: unknown): string =>
  typeof value === 'string' ? value : '';

const unescapeNewlines = (text: string): string =>
  text.replaceAll(String.raw`\n`, '\n');

const toStringChunks = async function* (
  source: SseSource,
): AsyncGenerator<string> {
  const decoder = new TextDecoder();

  if (source instanceof ReadableStream) {
    const reader = source.getReader();

    try {
      for (;;) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        yield decoder.decode(value, { stream: true });
      }

      const tail = decoder.decode();

      if (tail) {
        yield tail;
      }
    } finally {
      reader.releaseLock();
    }

    return;
  }

  for await (const chunk of source) {
    yield typeof chunk === 'string'
      ? chunk
      : decoder.decode(chunk, { stream: true });
  }

  const iterableTail = decoder.decode();

  if (iterableTail) {
    yield iterableTail;
  }
};

type Frame = { dataRaw: string; eventName: null | string };

const splitFrame = (frame: string): Frame => {
  let eventName: null | string = null;
  const dataLines: string[] = [];

  for (const rawLine of frame.split('\n')) {
    const line = rawLine.replace(TRAILING_CR, '');

    if (line === '' || line.startsWith(':')) {
      continue;
    }

    if (line.startsWith('event:')) {
      eventName = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).replace(LEADING_SPACE, ''));
    }
  }

  return { dataRaw: dataLines.join('\n'), eventName };
};

const buildEvent = (
  eventName: string,
  obj: Record<string, unknown>,
): null | ParsedEvent => {
  switch (eventName) {
    case 'done':
      return { type: 'done' };
    case 'error':
      return {
        code: toErrorCode(obj['code']),
        message: asString(obj['message']),
        type: 'error',
      };
    case 'reset':
      return { type: 'reset' };
    case 'status':
      return {
        label: asString(obj['label']),
        state: asString(obj['state']),
        type: 'status',
        ...(typeof obj['tool'] === 'string' && { tool: obj['tool'] }),
      };
    case 'token':
      return { text: asString(obj['text']), type: 'token' };
    default:
      return null;
  }
};

const parseFrame = (frame: string): null | ParsedEvent => {
  const { dataRaw, eventName } = splitFrame(frame);

  if (dataRaw.length === 0 && eventName === null) {
    return null;
  }

  if (eventName === null) {
    return { text: unescapeNewlines(dataRaw), type: 'token' };
  }

  let parsed: unknown = {};

  if (dataRaw.length > 0) {
    try {
      parsed = JSON.parse(dataRaw);
    } catch {
      if (eventName === 'token') {
        return { text: unescapeNewlines(dataRaw), type: 'token' };
      }

      parsed = {};
    }
  }

  return buildEvent(eventName, (parsed ?? {}) as Record<string, unknown>);
};

export const parseProtocolV2 = async function* (
  source: SseSource,
): AsyncGenerator<ParsedEvent, void, unknown> {
  let buffer = '';

  for await (const chunk of toStringChunks(source)) {
    buffer = (buffer + chunk).replaceAll('\r\n', '\n');

    let sepIndex = buffer.indexOf('\n\n');

    while (sepIndex !== -1) {
      const frame = buffer.slice(0, sepIndex);

      buffer = buffer.slice(sepIndex + 2);

      const ev = parseFrame(frame);

      if (ev) {
        yield ev;
      }

      sepIndex = buffer.indexOf('\n\n');
    }
  }

  const tail = buffer.replaceAll('\r\n', '\n').trim();

  if (tail) {
    const ev = parseFrame(tail);

    if (ev) {
      yield ev;
    }
  }
};

import 'server-only';
import {
  createResumableStreamContext,
  type ResumableStreamContext,
} from 'resumable-stream/redis';

import { RESUMABLE_STREAM_REDIS_URL } from '@/lib/env';

const STREAM_KEY_PREFIX = 'finki-hub-chat';
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu;

export type WaitUntil = (promise: Promise<unknown>) => void;

type ActiveProducerRegistry = {
  readonly abort: (streamId: string) => boolean;
  readonly has: (streamId: string) => boolean;
  readonly register: (streamId: string, controller: AbortController) => void;
  readonly unregister: (streamId: string) => void;
};

type CreateContextOptions = {
  readonly waitUntil?: WaitUntil;
};

export class InvalidResponseIdError extends Error {
  readonly headerName = 'X-Response-Id';

  readonly responseId: string;

  constructor(responseId: string, options?: ErrorOptions) {
    super('Python chat stream X-Response-Id must be a UUID.', options);
    this.name = 'InvalidResponseIdError';
    this.responseId = responseId;
  }
}

export class MissingResponseIdError extends Error {
  readonly headerName = 'X-Response-Id';

  constructor() {
    super('Python chat stream response is missing X-Response-Id.');
    this.name = 'MissingResponseIdError';
  }
}

export const normalizePythonResponseStreamId = (
  responseId: null | string | undefined,
): string => {
  if (
    responseId === null ||
    responseId === undefined ||
    responseId.length === 0
  ) {
    throw new MissingResponseIdError();
  }

  if (!UUID_PATTERN.test(responseId)) {
    throw new InvalidResponseIdError(responseId);
  }

  return responseId;
};

export const createChatResumableStreamContext = ({
  waitUntil,
}: CreateContextOptions = {}): ResumableStreamContext => {
  process.env['REDIS_URL'] = RESUMABLE_STREAM_REDIS_URL;

  return createResumableStreamContext({
    keyPrefix: STREAM_KEY_PREFIX,
    waitUntil: waitUntil ?? null,
  });
};

export const createActiveProducerRegistry = (): ActiveProducerRegistry => {
  const controllers = new Map<string, AbortController>();

  return {
    abort: (streamId) => {
      const controller = controllers.get(streamId);

      if (controller === undefined) {
        return false;
      }

      controller.abort();

      return true;
    },
    has: (streamId) => controllers.has(streamId),
    register: (streamId, controller) => {
      controllers.set(streamId, controller);
    },
    unregister: (streamId) => {
      controllers.delete(streamId);
    },
  };
};

export const activeChatProducers = createActiveProducerRegistry();

export const ACTIVE_PRODUCER_PROCESS_CRASH_LIMITATION =
  'Active producer AbortControllers are process-local; after a web process crash only Redis replay and Postgres state are recoverable.';

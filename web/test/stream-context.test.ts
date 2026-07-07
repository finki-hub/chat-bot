import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const VALID_RESPONSE_ID = '018f0f36-2b1d-7cc0-a50b-5f2d90c91d21';
const STREAM_REDIS_URL = 'redis://stream-store:6379';

type FakeStreamContext = {
  readonly marker: 'fake-resumable-context';
};

type StreamContextOptions = {
  readonly keyPrefix?: string;
  readonly waitUntil: ((promise: Promise<unknown>) => void) | null;
};

const mocks = vi.hoisted(() => {
  const createResumableStreamContext = vi.fn<
    (options: StreamContextOptions) => FakeStreamContext
  >(() => ({
    marker: 'fake-resumable-context',
  }));

  return { createResumableStreamContext };
});

vi.mock('resumable-stream/redis', () => ({
  createResumableStreamContext: mocks.createResumableStreamContext,
}));

const resetEnv = (): void => {
  process.env['API_BASE_URL'] = 'https://api:8880';
  process.env['CHAT_API_KEY'] = 'secret-key';
  process.env['RESUMABLE_STREAM_REDIS_URL'] = STREAM_REDIS_URL;
};

const getContextOptions = (): StreamContextOptions => {
  const call = mocks.createResumableStreamContext.mock.calls[0];

  if (call === undefined) {
    throw new Error('createResumableStreamContext was not called');
  }

  return call[0];
};

describe('resumable stream context utilities', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    resetEnv();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('creates a Redis-backed resumable stream context from server env', async () => {
    // Given: server-only Redis configuration is present.
    const waitUntil = vi.fn<(promise: Promise<unknown>) => void>();
    const { createChatResumableStreamContext } =
      await import('@/lib/resumable-stream-context');

    // When: a route asks for the resumable stream context.
    const context = createChatResumableStreamContext({ waitUntil });
    const secondContext = createChatResumableStreamContext({ waitUntil });

    // Then: the library's default Redis clients will read the private URL.
    expect(context).toStrictEqual({ marker: 'fake-resumable-context' });
    expect(secondContext).toBe(context);
    expect(process.env['REDIS_URL']).toBe(STREAM_REDIS_URL);
    expect(mocks.createResumableStreamContext).toHaveBeenCalledOnce();
    expect(getContextOptions()).toStrictEqual({
      keyPrefix: 'finki-hub-chat',
      waitUntil,
    });
  });

  it('normalizes stream id to exactly the Python X-Response-Id value', async () => {
    // Given: Python returned a UUID response id header.
    const { normalizePythonResponseStreamId } =
      await import('@/lib/resumable-stream-context');

    // When: the BFF normalizes it for resumable-stream.
    const streamId = normalizePythonResponseStreamId(VALID_RESPONSE_ID);

    // Then: the stream id is the response id, not a divergent BFF id.
    expect(streamId).toBe(VALID_RESPONSE_ID);
  });

  it('throws a typed error for a missing Python X-Response-Id', async () => {
    // Given: no producer was registered before normalization.
    const {
      activeChatProducers,
      MissingResponseIdError,
      normalizePythonResponseStreamId,
    } = await import('@/lib/resumable-stream-context');

    // When / Then: missing response id fails before any producer registration.
    expect(() => normalizePythonResponseStreamId(null)).toThrow(
      MissingResponseIdError,
    );
    expect(activeChatProducers.has(VALID_RESPONSE_ID)).toBe(false);
  });

  it('throws a typed error for a malformed Python X-Response-Id', async () => {
    // Given: Python returned a non-UUID response id.
    const { InvalidResponseIdError, normalizePythonResponseStreamId } =
      await import('@/lib/resumable-stream-context');

    // When / Then: invalid response ids fail before streaming starts.
    expect(() => normalizePythonResponseStreamId('resp-123')).toThrow(
      InvalidResponseIdError,
    );
  });

  it('tracks active producers for add, abort, and delete', async () => {
    // Given: an active Python producer has a registered AbortController.
    const { createActiveProducerRegistry } =
      await import('@/lib/resumable-stream-context');
    const registry = createActiveProducerRegistry();
    const controller = new AbortController();

    // When: the producer is registered, explicitly aborted, then unregistered.
    registry.register(VALID_RESPONSE_ID, controller);
    const aborted = registry.abort(VALID_RESPONSE_ID);
    registry.unregister(VALID_RESPONSE_ID);

    // Then: stop cancellation is best-effort and process-local.
    expect(aborted).toBe(true);
    expect(controller.signal.aborted).toBe(true);
    expect(registry.has(VALID_RESPONSE_ID)).toBe(false);
  });
});

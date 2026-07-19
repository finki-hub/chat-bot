import { get, type IncomingMessage } from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';
import { describe, expect, it } from 'vitest';

import { startChatStreamServer } from '../e2e/helpers/sse';

describe('startChatStreamServer', () => {
  it('closes while an SSE response is active', async () => {
    // Given: a client holds an in-progress SSE response open.
    const server = await startChatStreamServer({
      gapMs: 30_000,
      head: [{ type: 'start' }],
      tail: [],
    });
    const response = await new Promise<IncomingMessage>((resolve, reject) => {
      get(server.url, resolve).once('error', reject);
    });

    // When: test teardown closes the server before the stream completes.
    const closePromise = server.close();
    const result = await Promise.race([
      (async () => {
        await closePromise;
        return 'closed' as const;
      })(),
      delay(500, 'timed-out' as const, { ref: false }),
    ]);

    // Then: teardown finishes without waiting for the client to disconnect.
    response.destroy();
    await closePromise;

    expect(result).toBe('closed');
  });
});

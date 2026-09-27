import type { UIMessageChunk } from 'ai';

import { describe, expect, it } from 'vitest';

import { ActiveChatRequest } from '@/lib/active-chat-request';

const openRequest = async (
  active: ActiveChatRequest,
  metadata = active.begin(),
  signal?: AbortSignal,
) => {
  const source = new TransformStream<UIMessageChunk, UIMessageChunk>();
  const transport = active.transport({
    reconnectToStream: () => Promise.resolve(source.readable),
    sendMessages: () => Promise.resolve(source.readable),
  });
  const stream = await transport.sendMessages({
    abortSignal: signal,
    chatId: 'conversation',
    messageId: undefined,
    messages: [],
    metadata,
    trigger: 'submit-message',
  });
  const reader = stream.getReader();
  const writer = source.writable.getWriter();
  return {
    cancel: () => reader.cancel(),
    write: async (chunk: UIMessageChunk) => {
      const read = reader.read();
      await writer.write(chunk);
      return read;
    },
  };
};

describe('active request response identity', () => {
  it('clears before a new request and rejects a late start from the old epoch', async () => {
    const active = new ActiveChatRequest('conversation');
    const old = await openRequest(active);
    await old.write({
      messageMetadata: { responseId: 'old-response' },
      type: 'start',
    });
    active.begin();
    await old.write({
      messageMetadata: { responseId: 'late-old-response' },
      type: 'start',
    });

    expect(active.takeSnapshot()).toStrictEqual({ responseId: undefined });

    await old.cancel();
  });

  it('does not let old metadata, completion, or request cleanup overwrite the new response', async () => {
    const active = new ActiveChatRequest('conversation');
    const oldEpoch = active.begin();
    const old = await openRequest(active, oldEpoch);
    const current = await openRequest(active);
    await current.write({
      messageMetadata: { responseId: 'current-response' },
      type: 'message-metadata',
    });
    await old.write({
      messageMetadata: { responseId: 'old-response' },
      type: 'message-metadata',
    });
    await old.write({ type: 'finish' });
    active.finish(oldEpoch);

    expect(active.takeSnapshot()).toStrictEqual({
      responseId: 'current-response',
    });

    await old.cancel();
    await current.cancel();
  });

  it('does not revive an epoch stopped before its transport starts', async () => {
    const active = new ActiveChatRequest('conversation');
    const epoch = active.begin();

    expect(active.takeSnapshot()).toStrictEqual({ responseId: undefined });

    const stopped = await openRequest(active, epoch);
    await stopped.write({
      messageMetadata: { responseId: 'late-response' },
      type: 'start',
    });

    expect(active.takeSnapshot()).toBeUndefined();

    await stopped.cancel();
  });

  it.each<UIMessageChunk>([
    { type: 'finish' },
    { type: 'abort' },
    { errorText: 'failure', type: 'error' },
  ])(
    'clears at terminal frame $type and ignores later metadata',
    async (terminal) => {
      const active = new ActiveChatRequest('conversation');
      const current = await openRequest(active);
      await current.write({
        messageMetadata: { responseId: 'response' },
        type: 'start',
      });
      await current.write(terminal);
      await current.write({
        messageMetadata: { responseId: 'late-response' },
        type: 'message-metadata',
      });

      expect(active.takeSnapshot()).toBeUndefined();

      await current.cancel();
    },
  );

  it('clears on abort and ignores chunks delivered after cancellation', async () => {
    const active = new ActiveChatRequest('conversation');
    const abort = new AbortController();
    const current = await openRequest(active, active.begin(), abort.signal);
    await current.write({
      messageMetadata: { responseId: 'response' },
      type: 'start',
    });
    abort.abort();
    await current.write({
      messageMetadata: { responseId: 'late-response' },
      type: 'message-metadata',
    });

    expect(active.takeSnapshot()).toBeUndefined();

    await current.cancel();
  });

  it('seeds a resumed live stream only from its stream frame', async () => {
    const active = new ActiveChatRequest('conversation');
    const source = new TransformStream<UIMessageChunk, UIMessageChunk>();
    const transport = active.transport({
      reconnectToStream: () => Promise.resolve(source.readable),
      sendMessages: () => Promise.resolve(source.readable),
    });
    const stream = await transport.reconnectToStream({
      chatId: 'conversation',
    });
    if (stream === null) {
      throw new Error('Expected resumed stream');
    }
    const reader = stream.getReader();
    const read = reader.read();
    await source.writable.getWriter().write({
      messageMetadata: { responseId: 'resumed-response' },
      type: 'start',
    });
    await read;

    expect(active.takeSnapshot()).toStrictEqual({
      responseId: 'resumed-response',
    });

    await reader.cancel();
  });
});

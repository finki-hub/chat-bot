import type { UIMessageChunk } from 'ai';

import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ActiveChatRequest } from '@/lib/active-chat-request';
import { useStopChat } from '@/lib/use-stop-chat';

const mocks = vi.hoisted(() => ({
  capture:
    vi.fn<
      (event: string, properties: Readonly<Record<string, unknown>>) => void
    >(),
  stopServer: vi.fn<() => Promise<void>>(),
}));
vi.mock('posthog-js', () => ({ posthog: { capture: mocks.capture } }));
vi.mock('@/lib/transport', () => ({ stopChatStream: mocks.stopServer }));

describe('stop analytics linkage', () => {
  beforeEach(() => {
    mocks.capture.mockReset();
    mocks.stopServer.mockReset();
    mocks.stopServer.mockResolvedValue(undefined);
  });

  it.each(['opaque-response', undefined, 'https://private?key=secret'])(
    'captures a bounded response snapshot: %s',
    async (responseId) => {
      const stop = vi.fn<() => void>();
      const activeRequest = new ActiveChatRequest('conversation');
      const transport = activeRequest.transport({
        reconnectToStream: () => Promise.resolve(null),
        sendMessages: () =>
          Promise.resolve(
            new ReadableStream<UIMessageChunk>({
              start: (controller) => {
                controller.enqueue({
                  messageMetadata: { responseId },
                  type: 'start',
                });
              },
            }),
          ),
      });
      const stream = await transport.sendMessages({
        abortSignal: undefined,
        chatId: 'conversation',
        messageId: undefined,
        messages: [],
        trigger: 'submit-message',
      });
      const reader = stream.getReader();
      await reader.read();
      const { result } = renderHook(() =>
        useStopChat({
          activeRequest,
          convoIdRef: { current: 'conversation' },
          model: 'catalog-model',
          stop,
        }),
      );
      await act(async () => result.current());

      /* eslint-disable camelcase -- PostHog event property names. */
      expect(mocks.capture).toHaveBeenCalledWith('chat_stopped', {
        inference_model: 'catalog-model',
        ...(responseId === 'opaque-response' && { response_id: responseId }),
      });
      /* eslint-enable camelcase -- PostHog event property names. */
      expect(stop).toHaveBeenCalledOnce();

      expect(mocks.stopServer.mock.calls).toStrictEqual(
        responseId === 'opaque-response'
          ? [['conversation', { activeStreamId: responseId }]]
          : [['conversation', undefined]],
      );

      await reader.cancel();
    },
  );
});

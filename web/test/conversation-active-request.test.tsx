import { act, renderHook, waitFor } from '@testing-library/react';
import { useRef, useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { useConversationChatRuntime } from '@/lib/use-conversation-chat-runtime';
import { useConversationManagement } from '@/lib/use-conversation-management';
import { useStopChat } from '@/lib/use-stop-chat';

const capture = vi.hoisted(() =>
  vi.fn<
    (event: string, properties: Readonly<Record<string, unknown>>) => void
  >(),
);
vi.mock('posthog-js', () => ({
  posthog: {
    capture,
    /* eslint-disable camelcase -- SDK methods. */
    get_distinct_id: () => 'anonymous',
    get_session_id: () => 'session',
    /* eslint-enable camelcase -- SDK methods. */
  },
}));
vi.mock('@/lib/use-models', () => ({
  useModels: () => ({ models: [], refetch: async () => {} }),
}));
vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: () => ({
    hydratingConversation: false,
    retryHydration: () => {},
  }),
}));

const previous: MyUIMessage = {
  id: 'historical-sdk-message',
  metadata: { responseId: 'historical-response' },
  parts: [{ text: 'Previous answer', type: 'text' }],
  role: 'assistant',
};
const user: MyUIMessage = {
  id: 'new-user-message',
  parts: [{ text: 'New prompt', type: 'text' }],
  role: 'user',
};

const CURRENT_RESPONSE_ID = 'new-response';
const MODEL = 'catalog-model';
const requestUrl = (url: Parameters<typeof fetch>[0]): string =>
  url instanceof Request ? url.url : String(url);

const setup = () => {
  let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
  const request = vi.fn<typeof fetch>((url, init) => {
    if (requestUrl(url).endsWith('/stream')) {
      return Promise.resolve(new Response(null, { status: 204 }));
    }
    if (requestUrl(url).endsWith('/stop')) {
      return Promise.resolve(Response.json({ stopped: true }));
    }
    return Promise.resolve(
      new Response(
        new ReadableStream<Uint8Array>({
          start: (stream) => {
            controller = stream;
            init?.signal?.addEventListener(
              'abort',
              () => {
                stream.error(new DOMException('Aborted', 'AbortError'));
              },
              { once: true },
            );
          },
        }),
        { headers: { 'content-type': 'text/event-stream' } },
      ),
    );
  });
  vi.stubGlobal('fetch', request);
  const hook = renderHook(() => {
    const [activeId, setActiveId] = useState<null | string>('conversation');
    const preserveEmptyHydrationIdRef = useRef<null | string>(null);
    const runtime = useConversationChatRuntime({
      activeId,
      model: MODEL,
      preserveEmptyHydrationIdRef,
      reasoning: false,
      refreshConversations: async () => {},
      setActiveId,
    });
    const stop = useStopChat({
      activeRequest: runtime.activeRequest,
      convoIdRef: runtime.convoIdRef,
      model: MODEL,
      stop: runtime.stop,
    });
    const management = useConversationManagement({
      applyGeneratedTitle: async () => {},
      convoIdRef: runtime.convoIdRef,
      handleStop: stop,
      model: MODEL,
      preserveEmptyHydrationIdRef,
      refreshConversations: async () => {},
      sendMessageRef: runtime.sendMessageRef,
      setActiveError: runtime.setActiveError,
      setActiveId,
      setMessages: runtime.setMessages,
      status: runtime.status,
    });
    return { activeId, management, runtime, stop };
  });
  act(() => {
    hook.result.current.runtime.setMessages([previous]);
  });
  return {
    ...hook,
    close: () => controller?.close(),
    frame: (chunk: Readonly<Record<string, unknown>>) => {
      if (controller === undefined) {
        throw new Error('No current response');
      }
      controller.enqueue(
        new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`),
      );
    },
    request,
  };
};

describe('current request stop linkage with the installed AI SDK', () => {
  beforeEach(() => capture.mockReset());

  afterEach(() => vi.unstubAllGlobals());

  it('does not send a server stop for idle historical messages', async () => {
    const { request, result } = setup();
    // Allow the initial no-active-stream reconnect to settle.
    await act(async () => {});
    await act(async () => result.current.stop());

    expect(capture.mock.calls[0]?.[1]).not.toHaveProperty('response_id');
    expect(
      request.mock.calls.some(([url]) => requestUrl(url).endsWith('/stop')),
    ).toBe(false);
  });

  it.each(['new-chat', 'switch'] as const)(
    'stops the originating active conversation without an ID before %s',
    async (action) => {
      const { request, result } = setup();
      await act(async () => {});
      let pending = Promise.resolve();
      act(() => {
        pending = result.current.runtime.sendMessageRef.current(user);
      });
      await waitFor(() => {
        expect(request).toHaveBeenCalledWith('/api/chat', expect.anything());
        expect(result.current.runtime.status).toBe('submitted');
      });
      act(() => {
        if (action === 'new-chat') {
          result.current.management.handleNewChat();
        } else {
          result.current.management.handleSelect('next-conversation');
        }
      });
      await act(async () => pending);
      await waitFor(() => {
        expect(result.current.activeId).toBe(
          action === 'new-chat' ? null : 'next-conversation',
        );
      });

      expect(capture.mock.calls[0]?.[1]).not.toHaveProperty('response_id');
      expect(
        request.mock.calls.filter(([url]) => requestUrl(url).endsWith('/stop')),
      ).toStrictEqual([['/api/chat/conversation/stop', { method: 'POST' }]]);
    },
  );

  it.each(['submit', 'regenerate'] as const)(
    'clears synchronously before an immediate %s stop',
    async (kind) => {
      const { request, result } = setup();
      await act(async () => {
        const pending =
          kind === 'submit'
            ? result.current.runtime.sendMessageRef.current(user)
            : result.current.runtime.regenerate({ messageId: previous.id });
        await result.current.stop();
        await pending;
      });

      expect(capture.mock.calls[0]?.[1]).not.toHaveProperty('response_id');
      expect(request).toHaveBeenCalledWith('/api/chat/conversation/stop', {
        method: 'POST',
      });
    },
  );

  it.each(['submit', 'regenerate'] as const)(
    'never links historical response when stopping %s before start',
    async (kind) => {
      const { request, result } = setup();
      let pending = Promise.resolve();
      act(() => {
        pending =
          kind === 'submit'
            ? result.current.runtime.sendMessageRef.current(user)
            : result.current.runtime.regenerate({ messageId: previous.id });
      });
      await waitFor(() => {
        expect(request).toHaveBeenCalledWith('/api/chat', expect.anything());
      });
      await act(async () => result.current.stop());
      await act(async () => pending);

      /* eslint-disable camelcase -- PostHog event fields. */
      expect(capture).toHaveBeenCalledWith('chat_stopped', {
        inference_model: MODEL,
      });
      /* eslint-enable camelcase -- PostHog event fields. */
      expect(request).toHaveBeenCalledWith('/api/chat/conversation/stop', {
        method: 'POST',
      });
    },
  );

  it('uses only the new stream response ID for telemetry and the server guard', async () => {
    const { frame, request, result } = setup();
    let pending = Promise.resolve();
    act(() => {
      pending = result.current.runtime.sendMessageRef.current(user);
    });
    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('/api/chat', expect.anything());
    });
    act(() => {
      frame({
        messageId: 'new-sdk-message',
        messageMetadata: { responseId: CURRENT_RESPONSE_ID },
        type: 'start',
      });
      frame({ id: 'text', type: 'text-start' });
      frame({ delta: 'Partial answer', id: 'text', type: 'text-delta' });
    });
    await waitFor(() => {
      expect(result.current.runtime.messages.at(-1)?.metadata?.responseId).toBe(
        CURRENT_RESPONSE_ID,
      );
    });
    await act(async () => result.current.stop());
    await act(async () => pending);

    /* eslint-disable camelcase -- PostHog event fields. */
    expect(capture).toHaveBeenCalledWith('chat_stopped', {
      inference_model: MODEL,
      response_id: CURRENT_RESPONSE_ID,
    });
    /* eslint-enable camelcase -- PostHog event fields. */
    expect(request).toHaveBeenCalledWith(
      '/api/chat/conversation/stop',
      expect.objectContaining({
        body: JSON.stringify({ activeStreamId: CURRENT_RESPONSE_ID }),
      }),
    );
    expect(result.current.runtime.activeRequest.takeSnapshot()).toBeUndefined();
  });

  it('does not link a completed response retained in messages', async () => {
    const { close, frame, request, result } = setup();
    let pending = Promise.resolve();
    act(() => {
      pending = result.current.runtime.sendMessageRef.current(user);
    });
    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('/api/chat', expect.anything());
    });
    await act(async () => {
      frame({
        messageId: 'new-sdk-message',
        messageMetadata: { responseId: CURRENT_RESPONSE_ID },
        type: 'start',
      });
      frame({ type: 'finish' });
      close();
      await pending;
    });
    await act(async () => result.current.stop());

    expect(capture.mock.calls[0]?.[1]).not.toHaveProperty('response_id');
    expect(
      request.mock.calls.some(([url]) => requestUrl(url).endsWith('/stop')),
    ).toBe(false);
  });
});

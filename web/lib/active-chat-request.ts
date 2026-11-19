import type { ChatTransport, UIMessageChunk } from 'ai';

import type { MyUIMessage } from '@/lib/api-types';

const RESPONSE_ID_PATTERN = /^[\w-]{1,64}$/u;

class RequestEpoch {
  responseId: string | undefined;
}

// Local to one conversation runtime. History and SDK message IDs never seed it.
export class ActiveChatRequest {
  private current: RequestEpoch | undefined;

  constructor(readonly conversationId: null | string) {}

  begin(): RequestEpoch {
    const request = new RequestEpoch();
    this.current = request;
    return request;
  }

  finish(request: RequestEpoch): void {
    if (this.current === request) {
      this.current = undefined;
    }
  }

  async run<T>(operation: (request: RequestEpoch) => Promise<T>): Promise<T> {
    const request = this.begin();
    try {
      return await operation(request);
    } finally {
      this.finish(request);
    }
  }

  takeSnapshot(): undefined | { readonly responseId: string | undefined } {
    const snapshot =
      this.current === undefined
        ? undefined
        : { responseId: this.current.responseId };
    this.current = undefined;
    return snapshot;
  }

  transport(base: ChatTransport<MyUIMessage>): ChatTransport<MyUIMessage> {
    return {
      reconnectToStream: (options) =>
        this.track(
          this.begin(),
          () => base.reconnectToStream(options),
          options.abortSignal,
        ),
      sendMessages: async (options) => {
        const request =
          options.metadata instanceof RequestEpoch
            ? options.metadata
            : this.begin();
        // A stop can land before the SDK starts the request (regenerate awaits
        // its queue cleanup first). Reject as an abort so the SDK does not
        // finalize an empty assistant message for a request that never started.
        if (this.current !== request) {
          throw new DOMException(
            'Request stopped before transport start',
            'AbortError',
          );
        }
        const stream = await this.track(
          request,
          () => base.sendMessages(options),
          options.abortSignal,
        );
        if (stream === null) {
          throw new Error('Missing chat response stream');
        }
        return stream;
      },
    };
  }

  private observe(request: RequestEpoch, chunk: UIMessageChunk): void {
    if (this.current !== request) {
      return;
    }
    if (['abort', 'error', 'finish'].includes(chunk.type)) {
      this.finish(request);
      return;
    }
    if (chunk.type !== 'start' && chunk.type !== 'message-metadata') {
      return;
    }
    const metadata: unknown = chunk.messageMetadata;
    if (
      typeof metadata !== 'object' ||
      metadata === null ||
      !('responseId' in metadata)
    ) {
      return;
    }
    const responseId: unknown = metadata.responseId;
    if (
      typeof responseId === 'string' &&
      RESPONSE_ID_PATTERN.test(responseId)
    ) {
      request.responseId = responseId;
    }
  }

  private async track(
    request: RequestEpoch,
    open: () => Promise<null | ReadableStream<UIMessageChunk>>,
    signal?: AbortSignal,
  ): Promise<null | ReadableStream<UIMessageChunk>> {
    const finish = (): void => {
      this.finish(request);
      signal?.removeEventListener('abort', finish);
    };
    signal?.addEventListener('abort', finish, { once: true });
    if (signal?.aborted === true) {
      finish();
    }
    try {
      const stream = await open();
      if (stream === null) {
        finish();
        return null;
      }
      const reader = stream.getReader();
      return new ReadableStream<UIMessageChunk>({
        cancel: async () => {
          finish();
          await reader.cancel();
        },
        pull: async (controller) => {
          try {
            const result = await reader.read();
            if (result.done) {
              finish();
              controller.close();
              return;
            }
            this.observe(request, result.value);
            controller.enqueue(result.value);
          } catch (error) {
            finish();
            controller.error(error);
          }
        },
      });
    } catch (error) {
      finish();
      throw error;
    }
  }
}

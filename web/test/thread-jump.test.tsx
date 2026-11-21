import { act, render, waitFor } from '@testing-library/react';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  vi.spyOn(Element.prototype, 'scrollIntoView').mockReturnValue(undefined);
});

const MESSAGES: MyUIMessage[] = [
  {
    id: 'm1',
    metadata: {},
    parts: [{ text: 'Прашање', type: 'text' }],
    role: 'user',
  },
  {
    id: 'm2',
    metadata: {},
    parts: [{ text: 'Одговор', type: 'text' }],
    role: 'assistant',
  },
];

describe('Thread message jump', () => {
  it('scrolls to and highlights a search result message', async () => {
    const onJumpHandled = vi.fn<() => void>();
    render(
      <Thread
        jumpToMessageId="m2"
        messages={MESSAGES}
        onJumpHandled={onJumpHandled}
        status="ready"
      />,
    );

    const target = document.querySelector('[data-message-id="m2"]');

    await waitFor(() => {
      expect(target).toHaveClass('ring-primary/50');
    });

    // eslint-disable-next-line @typescript-eslint/unbound-method -- scrollIntoView is a test spy
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
    expect(onJumpHandled).toHaveBeenCalledOnce();
  });

  it('clears a jump when the message no longer exists', async () => {
    const onJumpHandled = vi.fn<() => void>();
    render(
      <Thread
        jumpToMessageId="missing"
        messages={MESSAGES}
        onJumpHandled={onJumpHandled}
        status="ready"
      />,
    );

    await waitFor(
      () => {
        expect(onJumpHandled).toHaveBeenCalledOnce();
      },
      { timeout: 3_000 },
    );
  });

  it('keeps the jump while the previous conversation is still shown', async () => {
    const onJumpHandled = vi.fn<() => void>();
    const previous: MyUIMessage[] = [
      {
        id: 'old',
        metadata: {},
        parts: [{ text: 'Старо', type: 'text' }],
        role: 'user',
      },
    ];
    const { rerender } = render(
      <Thread
        jumpToMessageId="m2"
        messages={previous}
        onJumpHandled={onJumpHandled}
        status="ready"
      />,
    );

    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 10);
      });
    });

    expect(onJumpHandled).not.toHaveBeenCalled();

    rerender(
      <Thread
        jumpToMessageId="m2"
        messages={MESSAGES}
        onJumpHandled={onJumpHandled}
        status="ready"
      />,
    );

    await waitFor(() => {
      expect(document.querySelector('[data-message-id="m2"]')).toHaveClass(
        'ring-primary/50',
      );
    });

    expect(onJumpHandled).toHaveBeenCalledOnce();
  });

  it('waits for hydration before resolving the jump', async () => {
    const onJumpHandled = vi.fn<() => void>();
    const { rerender } = render(
      <Thread
        hydrating
        jumpToMessageId="m2"
        messages={MESSAGES}
        onJumpHandled={onJumpHandled}
        status="ready"
      />,
    );

    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 10);
      });
    });

    expect(onJumpHandled).not.toHaveBeenCalled();

    rerender(
      <Thread
        hydrating={false}
        jumpToMessageId="m2"
        messages={MESSAGES}
        onJumpHandled={onJumpHandled}
        status="ready"
      />,
    );

    await waitFor(() => {
      expect(onJumpHandled).toHaveBeenCalledOnce();
    });
  });
});

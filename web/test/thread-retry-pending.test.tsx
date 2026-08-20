import { render, screen } from '@testing-library/react';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

const RETRY_LABEL = 'Обиди се повторно';

const messagesByPath: ReadonlyArray<readonly [string, MyUIMessage[]]> = [
  [
    'a preserved user message',
    [
      {
        id: 'user-message',
        metadata: {},
        parts: [{ text: 'Прашање', type: 'text' }],
        role: 'user',
      },
    ],
  ],
  [
    'a preserved assistant message',
    [
      {
        id: 'assistant-message',
        metadata: {},
        parts: [{ text: 'Претходен одговор', type: 'text' }],
        role: 'assistant',
      },
    ],
  ],
];

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

describe('Thread history retry pending state', () => {
  it.each(messagesByPath)('handles %s', (_name, messages) => {
    // Given a history failure retains messages while its retry is pending.
    render(
      <Thread
        activeError={{ code: 'history_load', message: 'request failed' }}
        messages={messages}
        onRetry={vi.fn<() => void>()}
        retryPending
        status="error"
      />,
    );

    // Then every rendered retry path prevents duplicate requests accessibly.
    const retry = screen.getByRole('button', { name: RETRY_LABEL });

    expect(retry).toBeDisabled();

    expect(retry).toHaveAttribute('aria-busy', 'true');
  });
});

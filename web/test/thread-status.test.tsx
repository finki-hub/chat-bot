import { render, screen } from '@testing-library/react';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

const USER_MESSAGE: MyUIMessage = {
  id: 'user-1',
  metadata: {},
  parts: [{ text: 'Кога се објавуваат резултатите?', type: 'text' }],
  role: 'user',
};
const ASSISTANT_SHELL: MyUIMessage = {
  id: 'assistant-1',
  metadata: {},
  parts: [],
  role: 'assistant',
};

describe('Thread pending status', () => {
  it('shows the active status while the SDK reports a ready assistant shell', () => {
    // Given transient retrieval status on an empty assistant shell reported as ready.
    render(
      <Thread
        activeStatus={{ label: '🔍 Пребарувам…', tool: 'search_documents' }}
        messages={[USER_MESSAGE, ASSISTANT_SHELL]}
        status="ready"
      />,
    );

    // Then the transient retrieval status remains visible before response content.
    expect(screen.getByTestId('search-status')).toHaveTextContent('Пребарувам');
    expect(screen.queryByTestId('typing-indicator')).not.toBeInTheDocument();
  });
});

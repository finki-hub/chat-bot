import { render, screen } from '@testing-library/react';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { AssistantMessage } from '@/components/chat/message';
import { SearchStatus } from '@/components/chat/search-status';
import { Thread } from '@/components/chat/thread';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

const SEARCHING = '🔍 Пребарувам…';
const PREAMBLE = 'Дозволете да проверам во базата…';
const ANSWER = 'Испитите се во јануари.';

const assistantWithParts = (parts: MyUIMessage['parts']): MyUIMessage => ({
  id: 'a1',
  metadata: {},
  parts,
  role: 'assistant',
});

const userMessage = (text: string): MyUIMessage => ({
  id: 'u1',
  metadata: {},
  parts: [{ text, type: 'text' }],
  role: 'user',
});

const hasText =
  (needle: string) =>
  (_content: string, element: Element | null): boolean => {
    if (element === null) {
      return false;
    }
    if (!element.textContent.includes(needle)) {
      return false;
    }
    return [...element.children].every(
      (child) => !child.textContent.includes(needle),
    );
  };

describe('SearchStatus', () => {
  it('renders the label and exposes a status role', () => {
    render(
      <SearchStatus
        label={SEARCHING}
        tool="faq_search"
      />,
    );
    const chip = screen.getByRole('status');

    expect(chip).toHaveTextContent(SEARCHING);
    expect(chip).toHaveAttribute('data-testid', 'search-status');
    expect(chip).toHaveAttribute('data-tool', 'faq_search');
  });

  it('renders without a tool', () => {
    render(<SearchStatus label="Размислувам…" />);

    expect(screen.getByRole('status')).toHaveTextContent('Размислувам…');
  });
});

describe('AssistantMessage', () => {
  it('renders only the LAST text part (preamble drop)', () => {
    const msg = assistantWithParts([
      { text: 'Барам во базата…', type: 'text' },
      { text: 'Конечниот одговор е тука.', type: 'text' },
    ]);
    render(<AssistantMessage message={msg} />);

    expect(screen.getByTestId('answer-text')).toHaveTextContent(
      'Конечниот одговор е тука.',
    );
    expect(
      screen.queryByText(hasText('Барам во базата…')),
    ).not.toBeInTheDocument();
  });

  it('shows the search chip when a status is active and no text yet', () => {
    const msg = assistantWithParts([]);
    render(
      <AssistantMessage
        message={msg}
        statusPart={{ label: SEARCHING }}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent(SEARCHING);
  });

  it('hides the search chip once answer text has arrived', () => {
    const msg = assistantWithParts([{ text: 'Одговор', type: 'text' }]);
    render(
      <AssistantMessage
        message={msg}
        statusPart={{ label: SEARCHING }}
      />,
    );

    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('suppresses a pre-tool preamble while streaming with an active status', () => {
    const msg = assistantWithParts([{ text: PREAMBLE, type: 'text' }]);
    render(
      <AssistantMessage
        message={msg}
        pending
        statusPart={{ label: SEARCHING }}
      />,
    );

    expect(screen.queryByText(hasText(PREAMBLE))).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent(SEARCHING);
  });

  it('shows the answer (not the preamble) once the answer part has started', () => {
    const msg = assistantWithParts([
      { text: PREAMBLE, type: 'text' },
      { text: ANSWER, type: 'text' },
    ]);
    render(
      <AssistantMessage
        message={msg}
        pending
        statusPart={{ label: SEARCHING }}
      />,
    );

    expect(screen.getByText(hasText(ANSWER))).toBeInTheDocument();
    expect(screen.queryByText(hasText(PREAMBLE))).not.toBeInTheDocument();
  });

  it('renders a Retry button for a non-interrupted error', () => {
    const onRetry = vi.fn<() => void>();
    const msg = assistantWithParts([]);
    render(
      <AssistantMessage
        errorPart={{ code: 'agent_error', message: 'Се случи грешка.' }}
        message={msg}
        onRetry={onRetry}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Се случи грешка.');

    screen.getByRole('button', { name: 'Обиди се повторно' }).click();

    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('shows a soft notice (no Retry) for an interrupted error', () => {
    const msg = assistantWithParts([{ text: 'Делумен одговор', type: 'text' }]);
    render(
      <AssistantMessage
        errorPart={{ code: 'interrupted', message: 'прекинато' }}
        message={msg}
      />,
    );

    expect(screen.getByTestId('answer-text')).toHaveTextContent(
      'Делумен одговор',
    );
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Одговорот е прекинат.',
    );
    expect(
      screen.queryByRole('button', { name: 'Обиди се повторно' }),
    ).not.toBeInTheDocument();
  });

  it('renders a timing footnote when finished with timing metadata', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: { timing: { totalMs: 2_345, ttftMs: 1_200 } },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(<AssistantMessage message={msg} />);

    const footnote = screen.getByTestId('message-timing');

    expect(footnote).toHaveTextContent('2.3s');
    expect(footnote).toHaveTextContent('прв токен');
    expect(footnote).toHaveTextContent('1.2s');
  });

  it('omits the first-token part when ttft is unknown', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: { timing: { totalMs: 500, ttftMs: null } },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(<AssistantMessage message={msg} />);

    const footnote = screen.getByTestId('message-timing');

    expect(footnote).toHaveTextContent('500ms');
    expect(footnote).not.toHaveTextContent('прв токен');
  });

  it('hides the timing footnote while pending', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: { timing: { totalMs: 2_345, ttftMs: 1_200 } },
      parts: [{ text: 'Делумен', type: 'text' }],
      role: 'assistant',
    };
    render(
      <AssistantMessage
        message={msg}
        pending
      />,
    );

    expect(screen.queryByTestId('message-timing')).not.toBeInTheDocument();
  });
});

describe('Thread', () => {
  it('shows the empty state when there are no messages', () => {
    render(
      <Thread
        messages={[]}
        status="ready"
      />,
    );

    expect(screen.getByText('Започни разговор')).toBeInTheDocument();
  });

  it('renders a user turn and an assistant answer (last text part)', () => {
    const messages: MyUIMessage[] = [
      userMessage('Кога е роковниот испит?'),
      assistantWithParts([
        { text: 'преамбула', type: 'text' },
        { text: 'Роковниот испит е во јануари.', type: 'text' },
      ]),
    ];
    render(
      <Thread
        messages={messages}
        status="ready"
      />,
    );

    expect(
      screen.getByText(hasText('Кога е роковниот испит?')),
    ).toBeInTheDocument();
    expect(screen.getByTestId('answer-text')).toHaveTextContent(
      'Роковниот испит е во јануари.',
    );
    expect(screen.queryByText(hasText('преамбула'))).not.toBeInTheDocument();
  });

  it('shows a typing indicator while awaiting the assistant reply', () => {
    render(
      <Thread
        messages={[userMessage('прашање')]}
        status="submitted"
      />,
    );

    expect(screen.getByTestId('typing-indicator')).toBeInTheDocument();
  });

  it('keeps the typing indicator mounted when an empty assistant stream shell appears', () => {
    const messages: MyUIMessage[] = [
      userMessage('прашање'),
      assistantWithParts([]),
    ];

    render(
      <Thread
        messages={messages}
        status="streaming"
      />,
    );

    expect(screen.getAllByTestId('typing-indicator')).toHaveLength(1);
    expect(screen.queryByTestId('answer-text')).not.toBeInTheDocument();
  });

  it('shows a live elapsed timer while awaiting the reply', () => {
    render(
      <Thread
        messages={[userMessage('прашање')]}
        status="submitted"
        streamStartedAt={Date.now()}
      />,
    );

    expect(screen.getByTestId('elapsed-timer')).toBeInTheDocument();
  });

  it('passes the active status only to the LAST assistant message while streaming', () => {
    const messages: MyUIMessage[] = [
      userMessage('прашање'),
      assistantWithParts([]),
    ];
    render(
      <Thread
        activeStatus={{ label: SEARCHING }}
        messages={messages}
        status="streaming"
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent(SEARCHING);
  });

  it('renders per-message actions via renderActions', () => {
    const messages: MyUIMessage[] = [
      assistantWithParts([{ text: 'готово', type: 'text' }]),
    ];
    render(
      <Thread
        messages={messages}
        renderActions={(m) => <span data-testid="acts">{m.id}</span>}
        status="ready"
      />,
    );

    expect(screen.getByTestId('acts')).toHaveTextContent('a1');
  });

  it('keeps completed message actions visible while a later answer streams', () => {
    const messages: MyUIMessage[] = [
      assistantWithParts([{ text: 'готово', type: 'text' }]),
      userMessage('уште едно'),
      { ...assistantWithParts([]), id: 'a2' },
    ];

    render(
      <Thread
        messages={messages}
        renderActions={(m) => <span data-testid="acts">{m.id}</span>}
        status="streaming"
      />,
    );

    expect(screen.getByTestId('acts')).toHaveTextContent('a1');
    expect(screen.getAllByTestId('typing-indicator')).toHaveLength(1);
  });
});

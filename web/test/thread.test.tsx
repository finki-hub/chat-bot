import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
const ANSWER_TEXT_TESTID = 'answer-text';
const TIMING_TESTID = 'message-timing';

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

  it('falls back to a tool-specific label when label is omitted', () => {
    render(<SearchStatus tool="faq_search" />);

    expect(screen.getByRole('status')).toHaveTextContent('Барам во прашања…');
  });

  it('falls back to the generic label for an unknown tool', () => {
    render(<SearchStatus tool="unknown_tool" />);

    expect(screen.getByRole('status')).toHaveTextContent('Пребарувам…');
  });
});

describe('AssistantMessage', () => {
  it('renders only the LAST text part (preamble drop)', () => {
    const msg = assistantWithParts([
      { text: 'Барам во базата…', type: 'text' },
      { text: 'Конечниот одговор е тука.', type: 'text' },
    ]);
    render(<AssistantMessage message={msg} />);

    expect(screen.getByTestId(ANSWER_TEXT_TESTID)).toHaveTextContent(
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

  it('streams a reset answer when the status has cleared and only the answer part remains', () => {
    const msg = assistantWithParts([{ text: ANSWER, type: 'text' }]);
    render(
      <AssistantMessage
        message={msg}
        pending
      />,
    );

    expect(screen.getByTestId(ANSWER_TEXT_TESTID)).toHaveTextContent(ANSWER);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('renders a stepper when the status has a stage', () => {
    const msg = assistantWithParts([]);
    render(
      <AssistantMessage
        message={msg}
        pending
        statusPart={{
          label: SEARCHING,
          stage: 'retrieve',
          tool: 'search_documents',
        }}
      />,
    );

    expect(screen.getByTestId('search-stepper')).toBeInTheDocument();
    expect(screen.getByText('Пребарување…')).toBeInTheDocument();
  });

  it('advances the stepper to the generate stage after the status resets', () => {
    const msg = assistantWithParts([]);
    const { rerender } = render(
      <AssistantMessage
        message={msg}
        pending
        statusPart={{
          label: SEARCHING,
          stage: 'context',
          tool: 'search_documents',
        }}
      />,
    );

    expect(screen.getByText('Составување…')).toBeInTheDocument();

    rerender(
      <AssistantMessage
        message={msg}
        pending
        statusPart={undefined}
      />,
    );

    expect(screen.getByText('Генерирање…')).toBeInTheDocument();
  });

  it('streams the answer in the no-reset pipeline path (a stage status stays active)', () => {
    // The retrieval pipeline keeps a `stage` status active through generation (no
    // reset is sent), so the real answer must stream under it rather than be
    // suppressed as a pre-tool preamble.
    const msg = assistantWithParts([{ text: ANSWER, type: 'text' }]);
    render(
      <AssistantMessage
        message={msg}
        pending
        statusPart={{ label: SEARCHING, stage: 'context' }}
      />,
    );

    expect(screen.getByTestId(ANSWER_TEXT_TESTID)).toHaveTextContent(ANSWER);
  });

  it('advances the stepper to generate when reasoning begins at the context stage', () => {
    const msg = assistantWithParts([
      { text: 'Размислувам…', type: 'reasoning' },
    ]);
    render(
      <AssistantMessage
        message={msg}
        pending
        statusPart={{ label: SEARCHING, stage: 'context' }}
      />,
    );

    expect(screen.getByText('Генерирање…')).toBeInTheDocument();
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

    expect(screen.getByTestId(ANSWER_TEXT_TESTID)).toHaveTextContent(
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

    const footnote = screen.getByTestId(TIMING_TESTID);

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

    const footnote = screen.getByTestId(TIMING_TESTID);

    expect(footnote).toHaveTextContent('500ms');
    expect(footnote).not.toHaveTextContent('прв токен');
  });

  it('renders the footnote as a hover trigger when diagnostics are present', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {
        diagnostics: {
          serverTotalMs: 1_234,
          tokens: { input: 10, output: 20, total: 30 },
        },
        timing: { totalMs: 2_345, ttftMs: 1_200 },
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(<AssistantMessage message={msg} />);

    const footnote = screen.getByTestId(TIMING_TESTID);

    expect(footnote.tagName).toBe('BUTTON');
    expect(footnote).toHaveTextContent('2.3s');
  });

  it('renders source cards for finished assistant answers', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {
        sources: [
          {
            id: 'q1',
            kind: 'faq',
            links: [{ label: 'iKnow', url: 'https://iknow.ukim.mk/' }],
            snippet: 'Уписот се прави преку iKnow.',
            title: 'Упис',
          },
          {
            chunkIndex: 4,
            id: 'c1',
            kind: 'chunk',
            section: 'Член 12',
            snippet: 'Правилата се наведени во членот.',
            title: 'Статут на ФИНКИ',
          },
        ],
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };

    render(<AssistantMessage message={msg} />);

    expect(screen.getByTestId('message-sources')).toHaveTextContent('Извори');
    expect(screen.getByText('Упис')).toBeInTheDocument();
    expect(screen.getByText('Статут на ФИНКИ · Член 12')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Врска: iKnow' })).toHaveAttribute(
      'href',
      'https://iknow.ukim.mk/',
    );
  });

  it('reveals the model and throughput rows in the diagnostics popover', async () => {
    const user = userEvent.setup();
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {
        // 30 output tokens over a 1.5s window → 20 tok/s.
        diagnostics: {
          serverTotalMs: 1_700,
          serverTtftMs: 200,
          tokens: { input: 10, output: 30, total: 40 },
        },
        inferenceModel: 'claude-sonnet-4-6',
        timing: { totalMs: 2_000, ttftMs: 200 },
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(<AssistantMessage message={msg} />);

    await user.hover(screen.getByTestId(TIMING_TESTID));

    await expect(screen.findByText('модел')).resolves.toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-4-6')).toBeInTheDocument();
    expect(screen.getByText('брзина (ток./сек)')).toBeInTheDocument();
    expect(screen.getByText('20')).toBeInTheDocument();
  });

  it('keeps the footnote a plain element when diagnostics are absent', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: { timing: { totalMs: 500, ttftMs: null } },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(<AssistantMessage message={msg} />);

    expect(screen.getByTestId(TIMING_TESTID).tagName).toBe('DIV');
  });

  it('keeps the timing footnote on a finished answer while a newer message streams', () => {
    // In the submit→first-token window the previous answer is briefly the last
    // assistant and receives pending; its timing is final and must stay visible.
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: { timing: { totalMs: 2_345, ttftMs: 1_200 } },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(
      <AssistantMessage
        message={msg}
        pending
      />,
    );

    expect(screen.getByTestId(TIMING_TESTID)).toBeInTheDocument();
  });

  it('shows no timing footnote on a message that is still generating', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {},
      parts: [{ text: 'Пишувам…', type: 'text' }],
      role: 'assistant',
    };
    render(
      <AssistantMessage
        message={msg}
        pending
      />,
    );

    expect(screen.queryByTestId(TIMING_TESTID)).toBeNull();
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
    expect(screen.getByTestId(ANSWER_TEXT_TESTID)).toHaveTextContent(
      'Роковниот испит е во јануари.',
    );
    expect(screen.queryByText(hasText('преамбула'))).not.toBeInTheDocument();
  });

  it('renders a persisted error notice after refresh (from message metadata)', () => {
    // After a refresh there is no live activeError; the errored turn is hydrated
    // with metadata.error and must still show the notice, not an empty bubble.
    const messages: MyUIMessage[] = [
      userMessage('прашање'),
      {
        id: 'a1',
        metadata: {
          error: { code: 'agent_error', message: 'Се случи грешка.' },
        },
        parts: [],
        role: 'assistant',
      },
    ];
    render(
      <Thread
        messages={messages}
        status="ready"
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Се случи грешка.');
  });

  it('prefers the live error over a stale persisted one on the last message', () => {
    const messages: MyUIMessage[] = [
      userMessage('прашање'),
      {
        id: 'a1',
        metadata: { error: { code: 'agent_error', message: 'Стара грешка.' } },
        parts: [],
        role: 'assistant',
      },
    ];
    render(
      <Thread
        activeError={{ code: 'agent_error', message: 'Грешка во живо.' }}
        messages={messages}
        status="ready"
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Грешка во живо.');
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
    expect(screen.queryByTestId(ANSWER_TEXT_TESTID)).not.toBeInTheDocument();
  });

  it('does not render an inline elapsed timer while awaiting the reply', () => {
    render(
      <Thread
        messages={[userMessage('прашање')]}
        status="submitted"
      />,
    );

    expect(screen.getByTestId('typing-indicator')).toBeInTheDocument();
    expect(screen.queryByTestId('elapsed-timer')).toBeNull();
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

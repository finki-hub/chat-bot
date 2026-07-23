import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { AssistantMessage, MessageError } from '@/components/chat/message';
import { SearchStatus } from '@/components/chat/search-status';
import { Thread } from '@/components/chat/thread';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

const SEARCHING = '🔍 Пребарувам…';
const PREAMBLE = 'Дозволете да проверам во базата…';
const ANSWER = 'Испитите се во јануари.';
const ARIA_EXPANDED = 'aria-expanded';
const ANSWER_TEXT_TESTID = 'answer-text';
const CHUNK_SOURCE_TITLE = 'Статут на ФИНКИ · Член 12';
const CHUNK_SOURCE_TITLE_RE = /Статут на ФИНКИ · Член 12/u;
const COLLAPSED_CHUNK_TEXT =
  'Правилата за запишување и заверка на семестарот се наведени во овој член…';
const EXPANDED_CHUNK_TEXT_RE = /Вториот став/u;
const FIRST_SOURCE_TITLE = 'Прв извор';
const SECOND_SOURCE_TITLE = 'Втор извор';
const THIRD_SOURCE_TITLE = 'Трет извор';
const SHOW_SOURCES_LABEL = 'Прикажи извори';
const HIDE_SOURCES_LABEL = 'Сокриј извори';
const SPONSORED_RESET_TEXT = '18 јул. 2026, 14:00.';
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

  it('renders safe generic guidance and Retry for an agent error', () => {
    const onRetry = vi.fn<() => void>();
    const msg = assistantWithParts([]);
    const rawMessage = 'provider secret: https://secret.invalid';

    render(
      <AssistantMessage
        errorPart={{ code: 'agent_error', message: rawMessage }}
        message={msg}
        onRetry={onRetry}
      />,
    );

    const alert = screen.getByRole('alert');

    expect(alert).toHaveTextContent(
      'Се случи неочекувана грешка. Обидете се повторно.',
    );
    expect(alert).not.toHaveTextContent(rawMessage);

    screen.getByRole('button', { name: 'Обиди се повторно' }).click();

    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('renders safe credential guidance and opens credential management', async () => {
    const onManageCredentials = vi.fn<() => void>();
    const rawMessage = 'provider detail: leaked credential state';
    const user = userEvent.setup();

    render(
      <MessageError
        errorPart={{ code: 'credential_required', message: rawMessage }}
        onManageCredentials={onManageCredentials}
        onRetry={vi.fn<() => void>()}
      />,
    );

    const alert = screen.getByRole('alert');

    expect(alert).toHaveTextContent('За избраниот модел е потребен API клуч.');
    expect(alert).not.toHaveTextContent(rawMessage);

    expect(
      screen.queryByRole('button', { name: 'Обиди се повторно' }),
    ).toBeNull();

    await user.click(screen.getByRole('button', { name: 'Додај API клуч' }));

    expect(onManageCredentials).toHaveBeenCalledOnce();
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

  it('renders safe sponsored quota guidance with one recovery action', async () => {
    const onManageCredentials = vi.fn<() => void>();
    const rawMessage = 'provider secret: https://attacker.invalid';
    const user = userEvent.setup();

    render(
      <MessageError
        errorPart={{
          code: 'free_quota_exhausted',
          message: rawMessage,
          // eslint-disable-next-line camelcase -- mirrors the SSE wire contract.
          resets_at: '2026-07-18T12:00:00Z',
        }}
        onManageCredentials={onManageCredentials}
      />,
    );

    const alert = screen.getByRole('alert');

    expect(alert).not.toHaveTextContent(rawMessage);
    expect(alert).not.toHaveTextContent('2026-07-18T12:00:00Z');
    expect(alert).toHaveTextContent('Бесплатната квота е искористена.');
    expect(alert).toHaveTextContent(SPONSORED_RESET_TEXT);

    expect(
      screen.getByRole('button', { name: 'Додај API клуч' }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Почекај до ресетирањето' }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Додај API клуч' }));

    expect(onManageCredentials).toHaveBeenCalledOnce();
  });

  it('formats sponsored quota rollover in Skopje time', () => {
    render(
      <MessageError
        errorPart={{
          code: 'free_quota_exhausted',
          message: 'safe detail',
          // eslint-disable-next-line camelcase -- mirrors the SSE wire contract.
          resets_at: '2026-07-18T22:30:00Z',
        }}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('19 јул. 2026, 00:30.');
  });

  it.each([
    [
      'free_tier_unavailable' as const,
      'Бесплатниот модел е привремено недостапен.',
    ],
    [
      'sponsored_request_in_progress' as const,
      'Барањето за бесплатниот модел е веќе во тек. Почекај да заврши.',
    ],
    [
      'no_answer' as const,
      'Не е пронајден одговор. Обидете се да го преформулирате прашањето.',
    ],
  ])(
    'renders safe guidance for %s without retry or raw backend text',
    (code, copy) => {
      const rawMessage = 'provider detail: do not show this';

      render(
        <MessageError
          errorPart={{ code, message: rawMessage }}
          onRetry={vi.fn<() => void>()}
        />,
      );

      const alert = screen.getByRole('alert');

      expect(alert).toHaveTextContent(copy);
      expect(alert).not.toHaveTextContent(rawMessage);

      expect(
        screen.queryByRole('button', { name: 'Обиди се повторно' }),
      ).toBeNull();
    },
  );

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

  it('opens a small source set by default and keeps its links touchable', async () => {
    const user = userEvent.setup();
    const chunkSnippet =
      'Правилата за запишување и заверка на семестарот се наведени во овој член со повеќе детали. Вториот став содржи дополнителни детали што треба да се прикажат кога картичката е отворена.';
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {
        sources: [
          {
            id: 'q1',
            kind: 'faq',
            links: [
              { label: 'iKnow', url: 'https://iknow.ukim.mk/' },
              { label: 'ФИНКИ', url: 'https://www.finki.ukim.mk/' },
            ],
            snippet: 'Уписот се прави преку iKnow.',
            title: 'Упис',
          },
          {
            chunkIndex: 4,
            id: 'c1',
            kind: 'chunk',
            links: [
              {
                label: 'Статут на ФИНКИ',
                url: 'https://raw.githubusercontent.com/finki-hub/documents/main/raw/statut_i_delovnik.pdf',
              },
            ],
            section: 'Член 12',
            snippet: chunkSnippet,
            title: 'Статут на ФИНКИ',
          },
        ],
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };

    render(<AssistantMessage message={msg} />);

    const sources = screen.getByTestId('message-sources');
    const toggle = screen.getByRole('button', { name: HIDE_SOURCES_LABEL });

    expect(sources).toHaveTextContent('Извори');
    expect(sources).toHaveTextContent('2');
    expect(toggle).toHaveAttribute(ARIA_EXPANDED, 'true');
    expect(toggle).toHaveClass('min-h-11', 'pointer-fine:min-h-0');
    expect(screen.getByText('Упис')).toBeInTheDocument();
    expect(screen.getByText(CHUNK_SOURCE_TITLE)).toBeInTheDocument();

    const chunkButton = screen.getByRole('button', {
      name: CHUNK_SOURCE_TITLE_RE,
    });
    const chunkText = screen.getByText(COLLAPSED_CHUNK_TEXT);

    expect(chunkButton).toHaveAttribute(ARIA_EXPANDED, 'false');
    expect(chunkText).toHaveClass('line-clamp-2');
    expect(screen.queryByText(EXPANDED_CHUNK_TEXT_RE)).toBeNull();

    await user.click(chunkButton);

    const expandedChunkText = screen.getByText(EXPANDED_CHUNK_TEXT_RE);

    expect(chunkButton).toHaveAttribute(ARIA_EXPANDED, 'true');
    expect(expandedChunkText).not.toHaveClass('line-clamp-2');
    expect(expandedChunkText).toHaveClass('whitespace-pre-wrap');

    const iKnowLink = screen.getByRole('link', { name: 'Врска: iKnow' });
    const finkiLink = screen.getByRole('link', { name: 'Врска: ФИНКИ' });
    const documentLink = screen.getByRole('link', {
      name: 'Врска: Статут на ФИНКИ',
    });

    expect(iKnowLink).toHaveAttribute('href', 'https://iknow.ukim.mk/');
    expect(iKnowLink).toHaveClass(
      'min-h-11',
      'min-w-11',
      'pointer-fine:min-h-0',
      'pointer-fine:min-w-0',
    );
    expect(finkiLink).toHaveAttribute('href', 'https://www.finki.ukim.mk/');
    expect(documentLink).toHaveAttribute(
      'href',
      'https://raw.githubusercontent.com/finki-hub/documents/main/raw/statut_i_delovnik.pdf',
    );

    await user.click(screen.getByRole('button', { name: HIDE_SOURCES_LABEL }));

    expect(
      screen.getByRole('button', { name: SHOW_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'false');
    expect(screen.queryByText('Упис')).toBeNull();
  });

  it('waits for answer completion before automatically opening a small source set', () => {
    const message: MyUIMessage = {
      id: 'a1',
      metadata: {
        sources: [
          { id: 'q1', kind: 'faq', title: FIRST_SOURCE_TITLE },
          { id: 'q2', kind: 'faq', title: SECOND_SOURCE_TITLE },
        ],
      },
      parts: [{ text: 'Одговорот сè уште се генерира.', type: 'text' }],
      role: 'assistant',
    };
    const view = render(
      <AssistantMessage
        message={message}
        pending
      />,
    );

    expect(
      screen.getByRole('button', { name: SHOW_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'false');

    view.rerender(<AssistantMessage message={message} />);

    expect(
      screen.getByRole('button', { name: HIDE_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'true');
  });

  it('keeps larger source sets collapsed until requested', () => {
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {
        sources: [
          { id: 'q1', kind: 'faq', title: FIRST_SOURCE_TITLE },
          { id: 'q2', kind: 'faq', title: SECOND_SOURCE_TITLE },
          { id: 'q3', kind: 'faq', title: THIRD_SOURCE_TITLE },
        ],
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };

    render(<AssistantMessage message={msg} />);

    expect(
      screen.getByRole('button', { name: SHOW_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'false');
    expect(screen.queryByText(FIRST_SOURCE_TITLE)).toBeNull();
  });

  it('keeps a larger source set collapsed when metadata arrives after text', () => {
    const emptyMessage: MyUIMessage = {
      id: 'a1',
      metadata: { sources: [] },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    const loadedMessage: MyUIMessage = {
      ...emptyMessage,
      metadata: {
        sources: [
          { id: 'q1', kind: 'faq', title: FIRST_SOURCE_TITLE },
          { id: 'q2', kind: 'faq', title: SECOND_SOURCE_TITLE },
          { id: 'q3', kind: 'faq', title: THIRD_SOURCE_TITLE },
        ],
      },
    };
    const view = render(<AssistantMessage message={emptyMessage} />);

    view.rerender(<AssistantMessage message={loadedMessage} />);

    expect(
      screen.getByRole('button', { name: SHOW_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'false');
    expect(screen.queryByText(FIRST_SOURCE_TITLE)).toBeNull();
  });

  it('collapses an automatically opened source set when it grows past two', () => {
    const smallMessage: MyUIMessage = {
      id: 'a1',
      metadata: {
        sources: [
          { id: 'q1', kind: 'faq', title: FIRST_SOURCE_TITLE },
          { id: 'q2', kind: 'faq', title: SECOND_SOURCE_TITLE },
        ],
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    const largeMessage: MyUIMessage = {
      ...smallMessage,
      metadata: {
        sources: [
          { id: 'q1', kind: 'faq', title: FIRST_SOURCE_TITLE },
          { id: 'q2', kind: 'faq', title: SECOND_SOURCE_TITLE },
          { id: 'q3', kind: 'faq', title: THIRD_SOURCE_TITLE },
        ],
      },
    };
    const view = render(<AssistantMessage message={smallMessage} />);

    expect(
      screen.getByRole('button', { name: HIDE_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'true');

    view.rerender(<AssistantMessage message={largeMessage} />);

    expect(
      screen.getByRole('button', { name: SHOW_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'false');
    expect(screen.queryByText(FIRST_SOURCE_TITLE)).toBeNull();
  });

  it('reveals the model, trace ID, and throughput rows in the diagnostics popover', async () => {
    const user = userEvent.setup();
    const msg: MyUIMessage = {
      id: 'a1',
      metadata: {
        // 30 output tokens over a 1.5s window → 20 tok/s.
        diagnostics: {
          cost: { inputUsd: 0.00003, outputUsd: 0.00045, totalUsd: 0.00048 },
          serverTotalMs: 1_700,
          serverTtftMs: 200,
          tokens: { input: 10, output: 30, total: 40 },
        },
        inferenceModel: 'claude-sonnet-4-6',
        responseId: '00000000-0000-4000-8000-000000000123',
        timing: { totalMs: 2_000, ttftMs: 200 },
      },
      parts: [{ text: 'Готово', type: 'text' }],
      role: 'assistant',
    };
    render(<AssistantMessage message={msg} />);

    await user.hover(screen.getByTestId(TIMING_TESTID));

    await expect(screen.findByText('модел')).resolves.toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-4-6')).toBeInTheDocument();
    expect(screen.getByText('trace ID')).toBeInTheDocument();
    expect(
      screen.getByText('00000000-0000-4000-8000-000000000123'),
    ).toBeInTheDocument();
    expect(screen.getByText('брзина (ток./сек)')).toBeInTheDocument();
    expect(screen.getByText('20')).toBeInTheDocument();
    expect(screen.getByText('цена')).toBeInTheDocument();
    expect(screen.getByText('$0.000480')).toBeInTheDocument();
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

describe('Thread source disclosure', () => {
  it('keeps completed sources open while waiting for the next reply', () => {
    const messages: MyUIMessage[] = [
      userMessage('Прво прашање'),
      {
        id: 'a1',
        metadata: {
          sources: [{ id: 'q1', kind: 'faq', title: FIRST_SOURCE_TITLE }],
        },
        parts: [{ text: 'Завршен одговор', type: 'text' }],
        role: 'assistant',
      },
      {
        id: 'u2',
        metadata: {},
        parts: [{ text: 'Следно прашање', type: 'text' }],
        role: 'user',
      },
    ];

    render(
      <Thread
        messages={messages}
        status="submitted"
      />,
    );

    expect(
      screen.getByRole('button', { name: HIDE_SOURCES_LABEL }),
    ).toHaveAttribute(ARIA_EXPANDED, 'true');
    expect(screen.getByText(FIRST_SOURCE_TITLE)).toBeInTheDocument();
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
          error: {
            code: 'agent_error',
            message: 'provider secret: persisted detail',
          },
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

    const alert = screen.getByRole('alert');

    expect(alert).toHaveTextContent(
      'Се случи неочекувана грешка. Обидете се повторно.',
    );
    expect(alert).not.toHaveTextContent('provider secret: persisted detail');
  });

  it('prefers the live error over a stale persisted one on the last message', () => {
    const messages: MyUIMessage[] = [
      userMessage('прашање'),
      {
        id: 'a1',
        metadata: {
          error: { code: 'no_answer', message: 'Стара грешка.' },
        },
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

    const alert = screen.getByRole('alert');

    expect(alert).toHaveTextContent(
      'Се случи неочекувана грешка. Обидете се повторно.',
    );
    expect(alert).not.toHaveTextContent('Не е пронајден одговор.');
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

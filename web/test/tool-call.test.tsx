import type { DynamicToolUIPart } from 'ai';

import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { AssistantMessage } from '@/components/chat/message';
import { Thread } from '@/components/chat/thread';
import { ToolCall } from '@/components/chat/tool-call';
import { parseChatStateMessages } from '@/lib/chat-history';
import { assistantState } from '@/lib/chat-route-state';
import { joinText } from '@/lib/message-parts';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

const TITLE = 'Пребарувам: упис';
const LABEL = 'Пребарување во базата на знаење';
const QUERY = 'упис';
const SUMMARY = 'Пронајдени 4 извори';
const PREAMBLE = 'Ќе проверам во базата.';
const ANSWER = 'Уписот е во јули.';
const TOOL_STATUS = { label: 'Пребарувам…', tool: 'search_knowledge_base' };
const RETRIEVE_STATUS = { ...TOOL_STATUS, stage: 'retrieve' };

const TOOL_CALL = 'tool-call';
const SUMMARY_ROW = 'tool-call-summary';
const PANEL = 'tool-call-panel';
const ANSWER_TEXT = 'answer-text';
const TYPING = 'activity-line';
const ACTIONS_STUB = 'answer-actions-stub';
const ACTIONS = <div data-testid={ACTIONS_STUB} />;
const DONE = 'output-available';
const FAILED = 'output-error';

const call = (
  toolCallId: string,
  state: 'input-available' | typeof DONE | typeof FAILED = 'input-available',
): DynamicToolUIPart => {
  const base = {
    input: { query: QUERY },
    title: TITLE,
    toolCallId,
    toolName: 'search_knowledge_base',
    type: 'dynamic-tool',
  } as const;

  if (state === DONE) {
    return { ...base, output: { summary: SUMMARY }, state };
  }

  return state === FAILED
    ? { ...base, errorText: 'Пребарувањето не успеа', state }
    : { ...base, state };
};

const toolStates = () =>
  screen.getAllByTestId(TOOL_CALL).map((node) => node.dataset['state']);

const assistant = (parts: MyUIMessage['parts']): MyUIMessage => ({
  id: 'a1',
  metadata: {},
  parts,
  role: 'assistant',
});

describe('AssistantMessage tool calls', () => {
  it('renders text and tool calls interleaved in stream order', () => {
    render(
      <AssistantMessage
        message={assistant([
          { text: PREAMBLE, type: 'text' },
          call('c1', DONE),
          { text: ANSWER, type: 'text' },
        ])}
      />,
    );

    const parts = screen.getByTestId('answer-parts');
    const order = [...parts.children].map((child) =>
      child instanceof HTMLElement ? (child.dataset['testid'] ?? '') : '',
    );

    expect(order).toStrictEqual([ANSWER_TEXT, 'tool-call', ANSWER_TEXT]);
    expect(parts).toHaveTextContent(PREAMBLE);
    expect(parts).toHaveTextContent(ANSWER);
  });

  it('starts collapsed and expands to show the query and summary', async () => {
    const user = userEvent.setup();
    render(<AssistantMessage message={assistant([call('c1', DONE)])} />);

    const toggle = screen.getByRole('button', { name: new RegExp(LABEL, 'u') });

    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId(PANEL)).not.toBeInTheDocument();

    await user.click(toggle);

    const panel = screen.getByTestId(PANEL);

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(panel).toHaveTextContent(QUERY);
    expect(panel).toHaveTextContent(SUMMARY);
  });

  it('shows running, done, error and interrupted states', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <AssistantMessage
        message={assistant([call('c1', DONE), call('c2')])}
        pending
      />,
    );
    const states = () =>
      screen.getAllByTestId(TOOL_CALL).map((node) => node.dataset['state']);

    expect(states()).toStrictEqual(['done', 'running']);
    expect(screen.getAllByTestId(TOOL_CALL)[1]).toHaveTextContent('Во тек');

    rerender(
      <AssistantMessage
        message={assistant([call('c1'), call('c2', FAILED)])}
      />,
    );
    await user.click(screen.getByTestId(SUMMARY_ROW));

    expect(states()).toStrictEqual(['incomplete', 'error']);
    expect(screen.getAllByTestId(TOOL_CALL)[1]).toHaveTextContent('Неуспешно');
  });

  it('does not show the tool status chip once tool parts exist', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1')])}
        pending
        statusPart={TOOL_STATUS}
      />,
    );

    expect(screen.queryByTestId('search-status')).not.toBeInTheDocument();
    expect(screen.getByTestId(TOOL_CALL)).toBeInTheDocument();
  });

  it('keeps pre-tool text visible while a tool status is active', () => {
    render(
      <AssistantMessage
        message={assistant([{ text: PREAMBLE, type: 'text' }, call('c1')])}
        pending
        statusPart={TOOL_STATUS}
      />,
    );

    expect(screen.getByTestId(ANSWER_TEXT)).toHaveTextContent(PREAMBLE);
  });

  it('shows the typing indicator while the model writes after resolved calls', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1', DONE)])}
        pending
        statusPart={TOOL_STATUS}
      />,
    );

    expect(screen.getByTestId(TYPING)).toHaveTextContent(
      'Читање на пронајдените извори…',
    );
  });

  it('labels calls by tool with the query inline, not by the API title', () => {
    render(<AssistantMessage message={assistant([call('c1', DONE)])} />);

    const row = screen.getByTestId(TOOL_CALL);

    // The full label shows from `sm` up; narrow screens get the short one.
    expect(within(row).getByText(LABEL)).toHaveClass('max-sm:hidden');
    expect(within(row).getByText('База на знаење')).toHaveClass('sm:hidden');
    expect(row).toHaveTextContent(`„${QUERY}“`);
    expect(row).not.toHaveTextContent(TITLE);
  });

  it('labels data tools with a tense-neutral phrase', () => {
    render(
      <AssistantMessage
        message={assistant([
          {
            input: {},
            state: 'input-available',
            title: 'Ги проверувам предметите',
            toolCallId: 'c1',
            toolName: 'list_courses',
            type: 'dynamic-tool',
          },
        ])}
      />,
    );

    expect(screen.getByTestId(TOOL_CALL)).toHaveTextContent(
      'Проверка на предметите',
    );
  });
});

describe('AssistantMessage tool-call progress', () => {
  it('stops spinning a call once answer text follows it', () => {
    const { rerender } = render(
      <AssistantMessage
        message={assistant([call('c1'), { text: ANSWER, type: 'text' }])}
        pending
      />,
    );

    expect(toolStates()).toStrictEqual(['incomplete']);

    rerender(
      <AssistantMessage
        message={assistant([call('c1'), call('c2', DONE)])}
        pending
      />,
    );

    // Parallel calls resolve in any order, so c1 is still running.
    expect(toolStates()).toStrictEqual(['running', 'done']);

    rerender(
      <AssistantMessage
        message={assistant([call('c1'), call('c2')])}
        pending
      />,
    );

    expect(toolStates()).toStrictEqual(['running', 'running']);
  });

  it('does not show the search stepper next to tool rows', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1')])}
        pending
        statusPart={RETRIEVE_STATUS}
      />,
    );

    expect(screen.queryByTestId('search-stepper')).not.toBeInTheDocument();
    expect(screen.queryByTestId(TYPING)).not.toBeInTheDocument();
  });

  it('shows the typing indicator, not the stepper, after calls resolve', () => {
    render(
      <AssistantMessage
        message={assistant([
          { text: PREAMBLE, type: 'text' },
          call('c1', DONE),
        ])}
        pending
        statusPart={RETRIEVE_STATUS}
      />,
    );

    expect(screen.queryByTestId('search-stepper')).not.toBeInTheDocument();
    expect(screen.getByTestId(TYPING)).toBeInTheDocument();
  });

  it('treats only text after the last call as the visible answer while streaming', () => {
    const { rerender } = render(
      <AssistantMessage
        actions={ACTIONS}
        message={assistant([
          { text: PREAMBLE, type: 'text' },
          call('c1', DONE),
        ])}
        pending
      />,
    );

    expect(screen.queryByTestId(ACTIONS_STUB)).not.toBeInTheDocument();

    rerender(
      <AssistantMessage
        actions={ACTIONS}
        message={assistant([{ text: PREAMBLE, type: 'text' }, call('c1')])}
        pending
      />,
    );

    expect(screen.queryByTestId(ACTIONS_STUB)).not.toBeInTheDocument();

    rerender(
      <AssistantMessage
        actions={ACTIONS}
        message={assistant([
          { text: PREAMBLE, type: 'text' },
          call('c1', DONE),
          { text: `\n\n${ANSWER}`, type: 'text' },
        ])}
        pending
      />,
    );

    expect(screen.getByTestId(ACTIONS_STUB)).toBeInTheDocument();
    expect(screen.queryByTestId(TYPING)).not.toBeInTheDocument();
  });

  it('shows the actions on a finished message that only has a preamble', () => {
    render(
      <AssistantMessage
        actions={ACTIONS}
        message={assistant([
          { text: PREAMBLE, type: 'text' },
          call('c1', DONE),
        ])}
      />,
    );

    expect(screen.getByTestId(ACTIONS_STUB)).toBeInTheDocument();
  });

  it('links the toggle to the panel only while it is open', async () => {
    const user = userEvent.setup();
    render(<AssistantMessage message={assistant([call('c1', DONE)])} />);

    const toggle = screen.getByRole('button', { name: new RegExp(LABEL, 'u') });

    expect(toggle).not.toHaveAttribute('aria-controls');

    await user.click(toggle);

    expect(toggle).toHaveAttribute(
      'aria-controls',
      screen.getByTestId(PANEL).id,
    );
  });

  it('omits the query row for a blank query', async () => {
    const user = userEvent.setup();
    render(
      <AssistantMessage
        message={assistant([{ ...call('c1', DONE), input: { query: '  ' } }])}
      />,
    );

    await user.click(
      screen.getByRole('button', { name: new RegExp(LABEL, 'u') }),
    );

    const panel = screen.getByTestId(PANEL);

    expect(panel).not.toHaveTextContent('Барање');
    expect(panel).toHaveTextContent(SUMMARY);
  });
});

const dataCall = (toolCallId: string): DynamicToolUIPart => ({
  input: {},
  output: { summary: '' },
  state: DONE,
  title: 'Предмети',
  toolCallId,
  toolName: 'list_courses',
  type: 'dynamic-tool',
});

const FIRST_LABEL = 'рокови';

const withSummary = (part: DynamicToolUIPart, summary: string) =>
  ({ ...part, output: { summary } }) as DynamicToolUIPart;

describe('AssistantMessage finished tool-call summary', () => {
  it('collapses several finished calls into one summary row', async () => {
    const user = userEvent.setup();
    render(
      <AssistantMessage
        message={assistant([
          { ...call('c1', DONE), input: { query: FIRST_LABEL } },
          call('c2', DONE),
          dataCall('c3'),
          { text: ANSWER, type: 'text' },
        ])}
      />,
    );

    const summary = screen.getByTestId(SUMMARY_ROW);

    expect(summary).toHaveTextContent('2 пребарувања · 1 проверка');
    expect(summary).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId(TOOL_CALL)).not.toBeInTheDocument();
    expect(screen.queryByTestId('tool-call-problems')).not.toBeInTheDocument();

    await user.click(summary);

    expect(summary).toHaveAttribute('aria-expanded', 'true');
    expect(toolStates()).toStrictEqual(['done', 'done', 'done']);

    await user.click(
      screen.getByRole('button', { name: new RegExp(FIRST_LABEL, 'u') }),
    );

    expect(screen.getByTestId(PANEL)).toHaveTextContent(SUMMARY);
  });

  it('keeps individual rows live while the message is pending', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1', DONE), call('c2', DONE), call('c3')])}
        pending
      />,
    );

    expect(screen.queryByTestId(SUMMARY_ROW)).not.toBeInTheDocument();
    expect(toolStates()).toStrictEqual(['done', 'done', 'running']);
  });

  it('keeps a single finished call as its own row', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1', DONE), { text: ANSWER, type: 'text' }])}
      />,
    );

    expect(screen.queryByTestId(SUMMARY_ROW)).not.toBeInTheDocument();
    expect(toolStates()).toStrictEqual(['done']);
  });

  it('counts failed and interrupted calls in the summary', () => {
    render(
      <AssistantMessage
        message={assistant([
          call('c1', DONE),
          call('c2', FAILED),
          call('c3'),
          call('c4', DONE),
        ])}
      />,
    );

    expect(screen.getByTestId(SUMMARY_ROW)).toHaveTextContent('4 пребарувања');
    expect(screen.getByTestId('tool-call-problems')).toHaveTextContent(
      '2 неуспешни или прекинати',
    );
  });

  it('uses the singular for one search among data tools', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1', DONE), dataCall('c2'), dataCall('c3')])}
      />,
    );

    expect(screen.getByTestId(SUMMARY_ROW)).toHaveTextContent(
      '1 пребарување · 2 проверки',
    );
  });

  it('collapses each run of calls separately around text', () => {
    render(
      <AssistantMessage
        message={assistant([
          call('c1', DONE),
          call('c2', DONE),
          { text: PREAMBLE, type: 'text' },
          call('c3', DONE),
          { text: ANSWER, type: 'text' },
        ])}
      />,
    );

    const order = [...screen.getByTestId('answer-parts').children].map(
      (child) =>
        child instanceof HTMLElement ? (child.dataset['testid'] ?? '') : '',
    );

    expect(order).toStrictEqual([
      'tool-call-group',
      ANSWER_TEXT,
      'tool-call',
      ANSWER_TEXT,
    ]);
  });
});

describe('ToolCall result count', () => {
  it.each([
    ['Пронајдени извори: 6', '6 резултати'],
    ['Пронајдени извори: 1', '1 резултат'],
    ['Нема пронајдени извори', 'нема резултати'],
    ['Пронајдени се делумно поврзани извори', 'делумно поврзани'],
  ])('shows %s as %s on the row header', (summary, badge) => {
    render(
      <AssistantMessage
        message={assistant([withSummary(call('c1', DONE), summary)])}
      />,
    );

    expect(screen.getByTestId('tool-call-count')).toHaveTextContent(badge);
  });

  it('shows no count for data tools or unfinished searches', () => {
    render(
      <AssistantMessage
        message={assistant([dataCall('c1'), call('c2')])}
        pending
      />,
    );

    expect(screen.queryByTestId('tool-call-count')).not.toBeInTheDocument();
  });

  it.each([
    [FAILED, 'Неуспешно', true],
    ['input-available', 'Прекинато', true],
    [DONE, 'Завршено', false],
  ] as const)('labels a %s call as %s', (state, label, visible) => {
    render(<ToolCall part={call('c1', state)} />);

    expect(screen.getByText(label).classList.contains('sr-only')).toBe(
      !visible,
    );
  });
});

describe('Thread tool calls', () => {
  it('shows a streaming assistant that only has tool calls so far', () => {
    render(
      <Thread
        messages={[
          {
            id: 'u1',
            metadata: {},
            parts: [{ text: 'Кога е уписот?', type: 'text' }],
            role: 'user',
          },
          assistant([call('c1')]),
        ]}
        status="streaming"
      />,
    );

    expect(
      within(screen.getByTestId(TOOL_CALL)).getByText(LABEL),
    ).toBeInTheDocument();
    expect(screen.queryByTestId(TYPING)).not.toBeInTheDocument();
  });

  it('shows one typing indicator inside the message once calls resolve', () => {
    render(
      <Thread
        messages={[
          {
            id: 'u1',
            metadata: {},
            parts: [{ text: 'Кога е уписот?', type: 'text' }],
            role: 'user',
          },
          assistant([call('c1', DONE)]),
        ]}
        status="streaming"
      />,
    );

    expect(screen.getAllByTestId(TYPING)).toHaveLength(1);
    expect(screen.getByTestId('answer-parts')).toBeInTheDocument();
  });
});

describe('tool part persistence', () => {
  it('round-trips tool parts while history text stays text only', async () => {
    const message = assistant([
      { text: PREAMBLE, type: 'text' },
      call('c1', DONE),
      call('c2', FAILED),
      { text: ANSWER, type: 'text' },
    ]);
    const { metadata, parts } = assistantState(message, {});
    const [restored] = await parseChatStateMessages([
      {
        content: joinText(message),
        id: message.id,
        metadata,
        parts,
        // eslint-disable-next-line camelcase -- chat-state wire field.
        response_id: null,
        role: 'assistant',
      },
    ]);

    expect(restored?.parts).toStrictEqual(message.parts);
    expect(joinText(message)).toBe(`${PREAMBLE}\n\n${ANSWER}`);
  });
});

describe('AssistantMessage layout across the stream', () => {
  it('folds a run as soon as the answer after it starts', () => {
    const parts: MyUIMessage['parts'] = [
      call('c1', DONE),
      call('c2', DONE),
      { text: ANSWER, type: 'text' },
    ];
    const view = render(
      <AssistantMessage
        message={assistant(parts)}
        pending
      />,
    );

    expect(screen.getByTestId(SUMMARY_ROW)).toHaveTextContent('2 пребарувања');

    // Finishing changes nothing about the tool rows.
    view.rerender(<AssistantMessage message={assistant(parts)} />);

    expect(screen.getByTestId(SUMMARY_ROW)).toHaveTextContent('2 пребарувања');
  });

  it('keeps sources and actions while a later call runs after answer text', () => {
    const message: MyUIMessage = {
      ...assistant([
        call('c1', DONE),
        { text: ANSWER, type: 'text' },
        call('c2'),
      ]),
      metadata: { sources: [{ id: 'q1', kind: 'faq', title: 'Упис' }] },
    };
    render(
      <AssistantMessage
        actions={ACTIONS}
        message={message}
        pending
      />,
    );

    expect(screen.getByTestId('message-sources')).toBeInTheDocument();
    expect(screen.getByTestId(ACTIONS_STUB)).toBeInTheDocument();
    expect(toolStates()).toStrictEqual(['done', 'running']);
  });

  it('folds finished reasoning into the run row once the answer starts', () => {
    const reasoning = {
      state: 'done',
      text: 'Размислувам',
      type: 'reasoning',
    } as const;
    const before = assistant([reasoning, call('c1', DONE), call('c2', DONE)]);
    const after = assistant([...before.parts, { text: ANSWER, type: 'text' }]);
    const tracked = new Set([
      'answer-text',
      'reasoning',
      SUMMARY_ROW,
      TOOL_CALL,
    ]);
    const order = (): string[] =>
      [
        ...screen
          .getByTestId('answer-parts')
          .querySelectorAll<HTMLElement>('[data-testid]'),
      ]
        .map((node) => node.dataset['testid'] ?? '')
        .filter((id) => tracked.has(id));
    const view = render(
      <AssistantMessage
        message={before}
        pending
      />,
    );

    expect(order()).toStrictEqual([TOOL_CALL, TOOL_CALL, 'reasoning']);

    const toggle = screen.getByTestId('reasoning');

    view.rerender(
      <AssistantMessage
        message={after}
        pending
      />,
    );

    expect(order()).toStrictEqual([SUMMARY_ROW, 'answer-text']);
    expect(toggle).not.toBeInTheDocument();
    expect(screen.getByTestId(SUMMARY_ROW)).toHaveTextContent(
      '2 пребарувања · размислување',
    );

    fireEvent.click(screen.getByTestId(SUMMARY_ROW));

    expect(screen.getByTestId('reasoning-panel')).toHaveTextContent(
      'Размислувам',
    );
  });

  it('folds a single finished call with the reasoning into one row', () => {
    render(
      <AssistantMessage
        message={assistant([
          { state: 'done', text: 'Размислувам', type: 'reasoning' },
          call('c1', DONE),
          { text: ANSWER, type: 'text' },
        ])}
      />,
    );

    expect(screen.queryByTestId('reasoning')).not.toBeInTheDocument();
    expect(screen.getByTestId(SUMMARY_ROW)).toHaveTextContent(
      '1 пребарување · размислување',
    );
  });

  it('keeps a single finished call as its row without reasoning', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1', DONE), { text: ANSWER, type: 'text' }])}
      />,
    );

    expect(screen.queryByTestId(SUMMARY_ROW)).not.toBeInTheDocument();
    expect(toolStates()).toStrictEqual(['done']);
  });
});

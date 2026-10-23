import type { DynamicToolUIPart } from 'ai';

import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { AssistantMessage } from '@/components/chat/message';
import { Thread } from '@/components/chat/thread';
import { parseChatStateMessages } from '@/lib/chat-history';
import { assistantState } from '@/lib/chat-route-state';
import { joinText } from '@/lib/message-parts';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

const LABEL = 'Пребарувам: упис';
const QUERY = 'упис';
const SUMMARY = 'Пронајдени 4 извори';
const PREAMBLE = 'Ќе проверам во базата.';
const ANSWER = 'Уписот е во јули.';
const TOOL_STATUS = { label: 'Пребарувам…', tool: 'search_knowledge_base' };

const TOOL_CALL = 'tool-call';
const DONE = 'output-available';
const FAILED = 'output-error';

const call = (
  toolCallId: string,
  state: 'input-available' | typeof DONE | typeof FAILED = 'input-available',
): DynamicToolUIPart => {
  const base = {
    input: { query: QUERY },
    title: LABEL,
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

    expect(order).toStrictEqual(['answer-text', 'tool-call', 'answer-text']);
    expect(parts).toHaveTextContent(PREAMBLE);
    expect(parts).toHaveTextContent(ANSWER);
  });

  it('starts collapsed and expands to show the query and summary', async () => {
    const user = userEvent.setup();
    render(<AssistantMessage message={assistant([call('c1', DONE)])} />);

    const toggle = screen.getByRole('button', { name: new RegExp(LABEL, 'u') });

    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId('tool-call-panel')).not.toBeInTheDocument();

    await user.click(toggle);

    const panel = screen.getByTestId('tool-call-panel');

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(panel).toHaveTextContent(QUERY);
    expect(panel).toHaveTextContent(SUMMARY);
  });

  it('shows running, done, error and interrupted states', () => {
    const { rerender } = render(
      <AssistantMessage
        message={assistant([call('c1'), call('c2', DONE)])}
        pending
      />,
    );
    const states = () =>
      screen.getAllByTestId(TOOL_CALL).map((node) => node.dataset['state']);

    expect(states()).toStrictEqual(['running', 'done']);
    expect(screen.getAllByTestId(TOOL_CALL)[0]).toHaveTextContent('Во тек');

    rerender(
      <AssistantMessage
        message={assistant([call('c1'), call('c2', FAILED)])}
      />,
    );

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

    expect(screen.getByTestId('answer-text')).toHaveTextContent(PREAMBLE);
  });

  it('shows the typing indicator while the model writes after resolved calls', () => {
    render(
      <AssistantMessage
        message={assistant([call('c1', DONE)])}
        pending
        statusPart={TOOL_STATUS}
      />,
    );

    expect(screen.getByTestId('typing-indicator')).toBeInTheDocument();
  });

  it('falls back to the tool label when the call has no title', () => {
    render(
      <AssistantMessage
        message={assistant([{ ...call('c1', DONE), title: '' }])}
      />,
    );

    expect(screen.getByTestId(TOOL_CALL)).toHaveTextContent(
      'Барам во базата на знаење…',
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
    expect(joinText(message)).toBe(`${PREAMBLE}${ANSWER}`);
  });
});

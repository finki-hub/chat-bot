import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { MyUIMessage, RetrievedSource } from '@/lib/api-types';

import { AssistantMessage } from '@/components/chat/message';
import { withSourceList } from '@/lib/source-groups';
import { parseProtocolV2 } from '@/lib/sse';

const FEES = 'Школарина за редовни студенти';
const RULES = 'Правилник за прв циклус';

const numbered: RetrievedSource[] = [
  { id: 'q1', kind: 'faq', ref: 1, title: FEES },
  { id: 'c1', kind: 'chunk', ref: 2, section: 'Член 12', title: RULES },
  { id: 'c2', kind: 'chunk', ref: 2, section: 'Член 14', title: RULES },
];

const answer = (
  text: string,
  sources: readonly RetrievedSource[],
): MyUIMessage => ({
  id: 'a1',
  metadata: { sources },
  parts: [{ text, type: 'text' }],
  role: 'assistant',
});

describe('inline citations', () => {
  it('turns [N] markers into chips for the numbered sources', () => {
    render(
      <AssistantMessage
        message={answer(
          'Изнесува 200 евра [1]. Важи и за вонредни [2][9].',
          numbered,
        )}
      />,
    );

    const chips = screen.getAllByTestId('citation');

    expect(chips.map((chip) => chip.textContent)).toStrictEqual(['1', '2']);
    expect(chips[0]).toHaveAccessibleName(`Извор 1: ${FEES}`);
    // A number the panel cannot show is dropped rather than left dangling.
    expect(screen.getByTestId('answer-text')).not.toHaveTextContent('[9]');
  });

  it('opens the sources and highlights the cited card', () => {
    render(<AssistantMessage message={answer('Член 12 [2].', numbered)} />);

    expect(screen.queryByTestId('source-panel')).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId('citation'));

    const panel = screen.getByTestId('source-panel');
    const card = within(panel).getByTestId('source-document');

    expect(card).toHaveAttribute('data-ref', '2');
    expect(card).toHaveClass('ring-2');
    expect(
      screen.getByRole('button', { name: 'Сокриј извори' }),
    ).toHaveAttribute('aria-expanded', 'true');
  });

  it('keeps the literal text when the sources carry no numbers', () => {
    render(
      <AssistantMessage
        message={answer('Види [1].', [{ id: 'q1', kind: 'faq', title: FEES }])}
      />,
    );

    expect(screen.queryByTestId('citation')).not.toBeInTheDocument();
    expect(screen.getByTestId('answer-text')).toHaveTextContent('Види [1].');
  });

  it('leaves brackets inside code alone', () => {
    render(
      <AssistantMessage
        message={answer('Пиши `items[1]` во кодот.', numbered)}
      />,
    );

    expect(screen.queryByTestId('citation')).not.toBeInTheDocument();
    expect(screen.getByText('items[1]')).toBeInTheDocument();
  });

  it('reads the citation number from the sources event', async () => {
    const frame = `event: sources
data: ${JSON.stringify({
      sources: [
        { id: 'q1', kind: 'faq', ref: 3, title: FEES },
        { id: 'q2', kind: 'faq', ref: -1, title: 'Друго' },
      ],
    })}

`;
    const events = await Array.fromAsync(
      parseProtocolV2({
        async *[Symbol.asyncIterator]() {
          yield await frame;
        },
      }),
    );

    expect(events).toStrictEqual([
      {
        sources: [
          { id: 'q1', kind: 'faq', ref: 3, title: FEES },
          { id: 'q2', kind: 'faq', title: 'Друго' },
        ],
        type: 'sources',
      },
    ]);
  });

  it('numbers the copied source list by citation', () => {
    expect(
      withSourceList(
        'Одговор [2][1].',
        [numbered[1], numbered[0]].filter((source) => source !== undefined),
        'Извори',
      ),
    ).toBe(`Одговор [2][1].\n\nИзвори:\n[1] ${FEES}\n[2] ${RULES}`);
  });
});

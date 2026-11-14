import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ReasoningDisclosure } from '@/components/chat/reasoning-disclosure';

const REASONING = 'thinking out loud';
const PANEL = 'reasoning-panel';

describe('ReasoningDisclosure', () => {
  it('stays collapsed while streaming, with a live label', () => {
    render(
      <ReasoningDisclosure
        streaming
        text={REASONING}
      />,
    );

    const button = screen.getByRole('button');

    expect(button).toHaveAttribute('aria-expanded', 'false');
    expect(button.querySelector('.text-shimmer')).not.toBeNull();
    expect(screen.queryByTestId(PANEL)).toBeNull();
  });

  it('stays collapsed once finished', () => {
    render(
      <ReasoningDisclosure
        streaming={false}
        text={REASONING}
      />,
    );

    expect(screen.getByRole('button')).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    expect(screen.queryByTestId(PANEL)).toBeNull();
  });

  it('opens and closes on click, also while streaming', () => {
    const view = render(
      <ReasoningDisclosure
        streaming
        text={REASONING}
      />,
    );
    const button = screen.getByRole('button');

    fireEvent.click(button);

    expect(button).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByTestId(PANEL)).toHaveTextContent(REASONING);

    // Finishing does not close a panel the user opened.
    view.rerender(
      <ReasoningDisclosure
        streaming={false}
        text={REASONING}
      />,
    );

    expect(screen.getByTestId(PANEL)).toBeInTheDocument();

    fireEvent.click(button);

    expect(screen.queryByTestId(PANEL)).toBeNull();
  });

  it('renders nothing when there is no reasoning text', () => {
    const { container } = render(
      <ReasoningDisclosure
        streaming
        text=""
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});

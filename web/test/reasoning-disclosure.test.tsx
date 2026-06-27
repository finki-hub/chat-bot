import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ReasoningDisclosure } from '@/components/chat/reasoning-disclosure';

const REASONING = 'thinking out loud';

describe('ReasoningDisclosure', () => {
  it('auto-expands so reasoning is visible live while streaming', () => {
    render(
      <ReasoningDisclosure
        streaming
        text={REASONING}
      />,
    );

    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByTestId('reasoning-panel')).toBeInTheDocument();
  });

  it('collapses once not streaming (answer started or done)', () => {
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
    expect(screen.queryByTestId('reasoning-panel')).toBeNull();
  });

  it('lets a manual toggle override the streaming default', () => {
    render(
      <ReasoningDisclosure
        streaming
        text={REASONING}
      />,
    );

    const button = screen.getByRole('button');

    expect(button).toHaveAttribute('aria-expanded', 'true');

    fireEvent.click(button);

    expect(button).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId('reasoning-panel')).toBeNull();
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

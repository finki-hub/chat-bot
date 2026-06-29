import type { ReactNode } from 'react';

import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ConversationScrollButton } from '@/components/ai-elements/conversation';
import { Message, MessageContent } from '@/components/ai-elements/message';

const stickToBottom = vi.hoisted(() => {
  const scrollToBottom = vi.fn<() => void>();
  // eslint-disable-next-line unicorn/consistent-function-scoping -- Vitest mock factories are hoisted, so mock components must be created inside vi.hoisted.
  const Content = ({ children }: { readonly children?: ReactNode }) => (
    <div>{children}</div>
  );
  // eslint-disable-next-line unicorn/consistent-function-scoping -- Vitest mock factories are hoisted, so mock components must be created inside vi.hoisted.
  const Root = ({ children }: { readonly children?: ReactNode }) => (
    <div>{children}</div>
  );
  Root.Content = Content;

  return { Root, scrollToBottom };
});

vi.mock('use-stick-to-bottom', () => ({
  StickToBottom: stickToBottom.Root,
  useStickToBottomContext: () => ({
    isAtBottom: false,
    scrollToBottom: stickToBottom.scrollToBottom,
  }),
}));

describe('ai-elements smoke', () => {
  beforeEach(() => {
    stickToBottom.scrollToBottom.mockClear();
  });

  it('renders a Message with text content', () => {
    render(
      <Message from="assistant">
        <MessageContent>здраво свет</MessageContent>
      </Message>,
    );

    expect(screen.getByText('здраво свет')).toBeInTheDocument();
  });

  it('exposes an accessible scroll-to-bottom button', () => {
    render(<ConversationScrollButton />);

    expect(
      screen.getByRole('button', { name: 'Скролувај до најновата порака' }),
    ).toBeInTheDocument();
  });
});

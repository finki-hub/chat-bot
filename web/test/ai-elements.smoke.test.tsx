import type { ReactNode } from 'react';

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  Conversation,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent } from '@/components/ai-elements/message';

type StickToBottomRootProps = {
  readonly children?: ReactNode;
  readonly initial?: boolean | ScrollBehavior;
  readonly resize?: ScrollBehavior;
};

const stickToBottom = vi.hoisted(() => {
  const rootProps: StickToBottomRootProps[] = [];
  const scrollToBottom = vi.fn<(behavior?: ScrollBehavior) => void>();
  // eslint-disable-next-line unicorn/consistent-function-scoping -- Vitest mock factories are hoisted, so mock components must be created inside vi.hoisted.
  const Content = ({ children }: { readonly children?: ReactNode }) => (
    <div>{children}</div>
  );
  const Root = ({ children, initial, resize }: StickToBottomRootProps) => {
    rootProps.push({ initial, resize });

    return <div>{children}</div>;
  };
  Root.Content = Content;

  return { Root, rootProps, scrollToBottom };
});

vi.mock('use-stick-to-bottom', () => ({
  StickToBottom: stickToBottom.Root,
  useStickToBottomContext: () => ({
    isAtBottom: false,
    scrollToBottom: stickToBottom.scrollToBottom,
  }),
}));

const createMediaQueryList = (
  matches: boolean,
  media: string,
): MediaQueryList => {
  const target = new EventTarget();

  return {
    addEventListener: target.addEventListener.bind(target),
    addListener() {},
    dispatchEvent: target.dispatchEvent.bind(target),
    matches,
    media,
    onchange: null,
    removeEventListener: target.removeEventListener.bind(target),
    removeListener() {},
  };
};

describe('ai-elements smoke', () => {
  beforeEach(() => {
    stickToBottom.rootProps.length = 0;
    stickToBottom.scrollToBottom.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
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

  it('uses instant scrolling when reduced motion is requested', async () => {
    vi.stubGlobal('matchMedia', (query: string) =>
      createMediaQueryList(query === '(prefers-reduced-motion: reduce)', query),
    );

    render(
      <Conversation>
        <div />
      </Conversation>,
    );
    render(<ConversationScrollButton />);

    await userEvent.click(
      screen.getByRole('button', { name: 'Скролувај до најновата порака' }),
    );

    expect(stickToBottom.rootProps.at(-1)).toMatchObject({
      initial: 'instant',
      resize: 'instant',
    });
    expect(stickToBottom.scrollToBottom).toHaveBeenCalledWith('instant');
  });
});

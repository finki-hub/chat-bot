import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { SharedChatScreen } from '@/components/chat/shared-chat-screen';

describe('SharedChatScreen', () => {
  it('shows when the conversation last changed and links to a new chat', () => {
    render(
      <SharedChatScreen
        messages={[]}
        title="Услови за дипломирање"
        updatedAt="2026-11-14T10:00:00Z"
      />,
    );

    expect(screen.getByTestId('shared-updated')).toHaveTextContent('2026');
    expect(
      screen.getByRole('link', { name: 'Постави свое прашање' }),
    ).toHaveAttribute('href', '/');
  });

  it('leaves the date out when it is unknown', () => {
    render(
      <SharedChatScreen
        messages={[]}
        title="Разговор"
      />,
    );

    expect(screen.queryByTestId('shared-updated')).toBeNull();
  });
});

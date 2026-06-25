import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Message, MessageContent } from '@/components/ai-elements/message';

describe('ai-elements smoke', () => {
  it('renders a Message with text content', () => {
    render(
      <Message from="assistant">
        <MessageContent>здраво свет</MessageContent>
      </Message>,
    );

    expect(screen.getByText('здраво свет')).toBeInTheDocument();
  });
});

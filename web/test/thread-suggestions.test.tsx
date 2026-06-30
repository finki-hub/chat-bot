import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { Thread } from '@/components/chat/thread';

const FIRST_SUGGESTION =
  'Колку кредити ми требаат за да се запишам во наредна година?';

describe('Thread suggestions', () => {
  it('shows suggestions when picking is available', () => {
    render(
      <Thread
        messages={[]}
        onPickSuggestion={vi.fn<(text: string) => void>()}
        status="ready"
      />,
    );

    expect(
      screen.getByRole('button', { name: FIRST_SUGGESTION }),
    ).toBeInTheDocument();
  });

  it('hides suggestions when picking is unavailable', () => {
    render(
      <Thread
        messages={[]}
        status="ready"
      />,
    );

    expect(
      screen.queryByRole('button', { name: FIRST_SUGGESTION }),
    ).not.toBeInTheDocument();
  });
});

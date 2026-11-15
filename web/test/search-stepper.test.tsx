import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { SearchStepper } from '@/components/chat/search-stepper';

describe('SearchStepper', () => {
  it('announces only the active stage in a live region', () => {
    render(<SearchStepper stages={['retrieve', 'rerank']} />);

    expect(screen.getByRole('status')).toHaveTextContent(
      'Избор на најрелевантните извори',
    );
    expect(screen.getByTestId('search-stepper')).not.toHaveAttribute(
      'aria-live',
    );
  });
});

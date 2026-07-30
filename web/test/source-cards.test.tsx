import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { RetrievedSource } from '@/lib/api-types';

import { SourceCards } from '@/components/chat/source-cards';

const SOURCE_TITLE = 'Study rules';
const MERMAID_CODE_RE = /flowchart LR/u;
const SOURCE_SNIPPET =
  '**Important** guidance with *expanded details* and a [reference link](https://example.com/).\n\n![tracking](https://example.com/pixel.png)\n\n```mermaid\nflowchart LR\nA@{ img: "https://evil.example/pixel.png" }\nclick A "https://evil.example/"\n```\n\n<picture><source srcset="https://evil.example/tracker.png"><img src="https://evil.example/fallback.png" alt="tracker"></picture>';
const SOURCE: RetrievedSource = {
  id: 'source-1',
  kind: 'chunk',
  snippet: SOURCE_SNIPPET,
  title: SOURCE_TITLE,
};

const renderSourceCards = () => {
  render(
    <SourceCards
      complete
      sources={[SOURCE]}
    />,
  );
};

describe('SourceCards', () => {
  it('renders source snippets as semantic Markdown without remote images', () => {
    renderSourceCards();

    expect(screen.getByRole('list')).toHaveClass('items-start');
    expect(screen.getByText('Important').tagName).toBe('STRONG');
    expect(screen.getByText(MERMAID_CODE_RE).closest('code')).not.toBeNull();
    expect(document.querySelectorAll('img[alt="tracking"]')).toHaveLength(0);
    expect(
      document.querySelectorAll('svg image, picture, source'),
    ).toHaveLength(0);
  });

  it('keeps collapsed Markdown non-interactive until expansion', async () => {
    const user = userEvent.setup();
    renderSourceCards();
    const cardButton = screen.getByRole('button', {
      name: new RegExp(SOURCE_TITLE, 'u'),
    });
    const snippetRegion = screen.getByText('Important').closest('[id]');

    expect(cardButton).toHaveClass('min-h-11', 'pointer-fine:min-h-0');
    expect(snippetRegion).toHaveAttribute('aria-hidden', 'true');
    expect(
      snippetRegion?.querySelectorAll(
        'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
      ),
    ).toHaveLength(0);
    expect(snippetRegion?.querySelectorAll('img')).toHaveLength(0);

    await user.click(cardButton);
    const markdownLink = screen.getByRole('button', { name: 'reference link' });

    expect(screen.getByText('expanded details').tagName).toBe('EM');
    expect(snippetRegion).not.toHaveAttribute('aria-hidden');
    expect(cardButton).not.toContainElement(markdownLink);

    await user.click(markdownLink);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(cardButton).toHaveAttribute('aria-expanded', 'true');
  });
});

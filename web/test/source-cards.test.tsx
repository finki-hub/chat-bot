import { render, screen, within } from '@testing-library/react';
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

const RULES = 'Правилник за студии';
const RULES_URL = 'https://finki.ukim.mk/rules.pdf';
const chunk = (
  id: string,
  extra: Partial<RetrievedSource> = {},
): RetrievedSource => ({
  authorityUrl: RULES_URL,
  id,
  kind: 'chunk',
  links: [{ label: RULES, url: RULES_URL }],
  snippet: `Текст ${id}`,
  title: RULES,
  ...extra,
});
const faq = (id: string): RetrievedSource => ({
  id,
  kind: 'faq',
  snippet: `Одговор ${id}`,
  title: 'Упис',
});

describe('SourceCards grouping', () => {
  it('groups chunks of one document into a card in first-appearance order', async () => {
    const user = userEvent.setup();
    render(
      <SourceCards
        complete
        sources={[
          faq('q1'),
          chunk('c1', { chunkIndex: 30, section: 'Член 26' }),
          faq('q2'),
          chunk('c2', { chunkIndex: 31, section: 'Член 27' }),
          chunk('c3', { chunkIndex: 32, section: 'Член 26' }),
        ]}
      />,
    );

    const toggle = screen.getByRole('button', { name: 'Прикажи извори' });

    expect(toggle).toHaveTextContent('3');

    await user.click(toggle);
    const panel = toggle.nextElementSibling;

    expect(panel).toHaveAttribute('id', toggle.getAttribute('aria-controls'));
    expect(
      [...(panel?.children ?? [])].map((item) => item.textContent),
    ).toStrictEqual([
      expect.stringContaining('Упис'),
      expect.stringContaining(RULES),
      expect.stringContaining('Упис'),
    ]);

    const card = screen.getByTestId('source-document');
    const sections = within(card).getByRole('list', { name: 'Делови' });

    expect(
      within(sections)
        .getAllByRole('listitem')
        .map((item) => item.textContent),
    ).toStrictEqual(['Член 26', 'Член 27']);
    expect(
      within(card).getAllByRole('link', { name: `Врска: ${RULES}` }),
    ).toHaveLength(1);
    expect(within(card).getAllByText(RULES)).toHaveLength(1);
  });

  it('expands a grouped document to every chunk snippet', async () => {
    const user = userEvent.setup();
    render(
      <SourceCards
        complete
        sources={[chunk('c1', { section: 'Член 26' }), chunk('c2')]}
      />,
    );

    const card = screen.getByTestId('source-document');
    const toggle = within(card).getByRole('button');

    expect(
      screen.getByRole('button', { name: 'Сокриј извори' }),
    ).toHaveTextContent('1');
    expect(within(card).queryByText('Текст c1')).not.toBeInTheDocument();

    await user.click(toggle);

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(within(card).getByText('Текст c1')).toBeInTheDocument();
    expect(within(card).getByText('Текст c2')).toBeInTheDocument();
  });

  it('counts parts when the chunks have no sections', () => {
    render(
      <SourceCards
        complete
        sources={[
          chunk('c1', { links: [] }),
          chunk('c2', { links: [] }),
          chunk('c3', { links: [] }),
        ]}
      />,
    );

    const card = screen.getByTestId('source-document');

    expect(card).toHaveTextContent('3 делови');
    expect(
      within(card).getByRole('link', { name: `Врска: ${RULES}` }),
    ).toHaveAttribute('href', RULES_URL);
  });

  it('keeps a single chunk per document as a plain card', () => {
    render(
      <SourceCards
        complete
        sources={[chunk('c1', { section: 'Член 26' }), faq('q1')]}
      />,
    );

    expect(screen.queryByTestId('source-document')).not.toBeInTheDocument();
    expect(screen.getByText(`${RULES} · Член 26`)).toBeInTheDocument();
  });
});

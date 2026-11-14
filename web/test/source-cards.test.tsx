import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { RetrievedSource } from '@/lib/api-types';

import { MessageFooter } from '@/components/chat/message-footer';
import {
  formatDocumentDate,
  isOutdatedSource,
  sourceLinkText,
} from '@/lib/source-groups';

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

// The sources live in the message footer: a toggle in its row, cards below.
const SourceCards = ({ sources }: { sources: readonly RetrievedSource[] }) => (
  <MessageFooter
    actions={null}
    diagnostics={undefined}
    sources={sources}
    timing={undefined}
  />
);

const openSources = () => {
  fireEvent.click(screen.getByRole('button', { name: 'Прикажи извори' }));
};

const renderSourceCards = () => {
  render(<SourceCards sources={[SOURCE]} />);
  openSources();
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
    const panel = document.querySelector(
      `#${CSS.escape(toggle.getAttribute('aria-controls') ?? '')}`,
    );

    expect(panel).toHaveAttribute('data-testid', 'source-panel');
    expect(
      [...(panel?.querySelector('ul')?.children ?? [])].map(
        (item) => item.textContent,
      ),
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
        sources={[chunk('c1', { section: 'Член 26' }), chunk('c2')]}
      />,
    );
    openSources();

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
        sources={[
          chunk('c1', { links: [] }),
          chunk('c2', { links: [] }),
          chunk('c3', { links: [] }),
        ]}
      />,
    );
    openSources();

    const card = screen.getByTestId('source-document');

    expect(card).toHaveTextContent('3 делови');
    expect(
      within(card).getByRole('link', { name: `Врска: ${RULES}` }),
    ).toHaveAttribute('href', RULES_URL);
  });

  it('keeps a single chunk per document as a plain card', () => {
    render(
      <SourceCards
        sources={[chunk('c1', { section: 'Член 26' }), faq('q1')]}
      />,
    );
    openSources();

    expect(screen.queryByTestId('source-document')).not.toBeInTheDocument();
    expect(screen.getByText(`${RULES} · Член 26`)).toBeInTheDocument();
  });
});

describe('source metadata', () => {
  it.each([
    ['2020-12-31', '31.12.2020'],
    ['2019-06', '06.2019'],
    ['2024', '2024'],
    ['unresolved', null],
    [undefined, null],
  ])('formats document date %s as %s', (value, expected) => {
    expect(formatDocumentDate(value)).toBe(expected);
  });

  it('flags only superseded or stale documents as outdated', () => {
    expect(
      isOutdatedSource(chunk('a', { currentStatus: 'likely_superseded' })),
    ).toBe(true);
    expect(
      isOutdatedSource(chunk('b', { currentStatus: 'stale_review_required' })),
    ).toBe(true);
    expect(
      isOutdatedSource(chunk('c', { currentStatus: 'currentness_unresolved' })),
    ).toBe(false);
    expect(isOutdatedSource(chunk('d'))).toBe(false);
  });

  it.each([
    [{ label: RULES, url: RULES_URL }, 'PDF · finki.ukim.mk'],
    [
      {
        label: RULES,
        url: 'https://raw.githubusercontent.com/finki-hub/documents/main/raw/rules.pdf',
      },
      'PDF · GitHub',
    ],
    [{ label: 'iKnow', url: 'https://iknow.ukim.mk/' }, 'iKnow'],
    [
      { label: RULES, url: 'https://www.finki.ukim.mk/mk/rules' },
      'finki.ukim.mk',
    ],
    [{ label: 'Врска', url: 'not a url' }, 'Врска'],
  ])('labels link %o as %s', (link, expected) => {
    expect(sourceLinkText(link, RULES)).toBe(expected);
  });

  it('shows the kind, the document date and an outdated warning on the card', () => {
    render(
      <SourceCards
        sources={[
          chunk('c1', {
            currentStatus: 'likely_superseded',
            documentDate: '2017-10-17',
          }),
        ]}
      />,
    );
    openSources();

    expect(screen.getByText('Документ')).toBeInTheDocument();
    expect(screen.getByText('17.10.2017')).toHaveAttribute(
      'datetime',
      '2017-10-17',
    );
    expect(screen.getByTestId('source-outdated')).toHaveTextContent(
      'Можеби застарено',
    );
    expect(screen.getByTestId('source-outdated-note')).toHaveTextContent(
      'Документот можеби е заменет со понов.',
    );
    expect(
      screen.getByRole('link', { name: `Врска: ${RULES}` }),
    ).toHaveTextContent('PDF');
  });

  it('shows only a compact count until the sources are opened', async () => {
    const user = userEvent.setup();
    render(<SourceCards sources={[faq('q1'), chunk('c1'), chunk('c2')]} />);

    const toggle = screen.getByRole('button', { name: 'Прикажи извори' });

    expect(toggle).toHaveTextContent('Извори2');
    expect(screen.queryByText(RULES)).not.toBeInTheDocument();
    expect(screen.queryByTestId('source-panel')).not.toBeInTheDocument();

    await user.click(toggle);

    expect(screen.getByTestId('source-panel')).toHaveTextContent(RULES);
  });
});

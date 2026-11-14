import { describe, expect, it } from 'vitest';

import type { RetrievedSource } from '@/lib/api-types';

import {
  citationNumbering,
  renumberSources,
  renumberText,
} from '@/lib/citations';

const sources: RetrievedSource[] = [
  { id: 'a', kind: 'faq', ref: 1, title: 'A' },
  { id: 'b', kind: 'faq', ref: 2, title: 'B' },
  { id: 'c', kind: 'chunk', ref: 4, title: 'C' },
  { id: 'd', kind: 'chunk', ref: 4, title: 'C' },
  { id: 'e', kind: 'faq', title: 'E' },
];

describe('citation numbering', () => {
  it('numbers sources in the order the answer cites them', () => {
    const numbering = citationNumbering('Прво [4]. Потоа [2][4].', sources);

    // Uncited sources follow, keeping their retrieval order.
    expect([...numbering]).toStrictEqual([
      [4, 1],
      [2, 2],
      [1, 3],
    ]);
    expect(renumberSources(sources, numbering).map((s) => s.ref)).toStrictEqual(
      [3, 2, 1, 1, undefined],
    );
  });

  it('ignores markers inside code and numbers no source carries', () => {
    const numbering = citationNumbering('`arr[2]` и [9] но [4].', sources);

    expect(numbering.get(4)).toBe(1);
    expect(numbering.get(9)).toBeUndefined();
  });

  it('rewrites the copied text to the shown numbers', () => {
    const text = 'Прво [4]. `x[4]` Потоа [2][9].\n```\ny[2]\n```';
    const numbering = citationNumbering(text, sources);

    expect(renumberText(text, numbering)).toBe(
      'Прво [1]. `x[4]` Потоа [2].\n```\ny[2]\n```',
    );
  });

  it('leaves answers without numbered sources alone', () => {
    const numbering = citationNumbering('Текст [1].', [
      { id: 'x', kind: 'faq', title: 'X' },
    ]);

    expect(numbering.size).toBe(0);
    expect(renumberText('Текст [1].', numbering)).toBe('Текст [1].');
  });
});

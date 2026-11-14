import type { RetrievedSource } from '@/lib/api-types';

// A run of `[N]` markers, e.g. "[2]" or "[1][3]". Only text nodes are scanned,
// so code spans, code blocks and link labels keep their brackets.
const CITATION_RUN_RE = /(?:\[\d{1,3}\])+/gu;
const CITATION_NUMBER_RE = /\d+/gu;

export const CITATION_TAG = 'cite-ref';

type MarkdownNode = {
  children?: MarkdownNode[];
  data?: { hName?: string; hProperties?: Record<string, string> };
  type: string;
  value?: string;
};

const citationNode = (ref: string): MarkdownNode => ({
  // A void element: the chip renders the number itself.
  children: [],
  data: { hName: CITATION_TAG, hProperties: { dataRef: ref } },
  type: 'citation',
});

const splitText = (value: string): MarkdownNode[] | null => {
  const parts: MarkdownNode[] = [];
  let last = 0;

  for (const match of value.matchAll(CITATION_RUN_RE)) {
    const start = match.index;

    if (start > last) {
      parts.push({ type: 'text', value: value.slice(last, start) });
    }

    const refs = match[0].matchAll(CITATION_NUMBER_RE);

    for (const [ref] of refs) {
      parts.push(citationNode(ref));
    }

    last = start + match[0].length;
  }

  if (parts.length === 0) {
    return null;
  }

  if (last < value.length) {
    parts.push({ type: 'text', value: value.slice(last) });
  }

  return parts;
};

const SKIPPED = new Set(['code', 'inlineCode', 'link', 'linkReference']);

const transform = (node: MarkdownNode): void => {
  if (node.children === undefined || SKIPPED.has(node.type)) {
    return;
  }

  node.children = node.children.flatMap((child) => {
    if (child.type === 'text' && typeof child.value === 'string') {
      return splitText(child.value) ?? [child];
    }

    transform(child);
    return [child];
  });
};

export const remarkCitations = () => transform;

// The source card a citation number points at: the first source carrying it.
export const sourceForRef = (
  sources: readonly RetrievedSource[],
  ref: number,
): RetrievedSource | undefined => sources.find((source) => source.ref === ref);

// Numbers a group of sources answers to (a document's chunks share one).
export const groupRef = (
  sources: readonly RetrievedSource[],
): number | undefined =>
  sources.find((source) => source.ref !== undefined)?.ref;

// Fenced blocks and inline code keep their brackets when numbering.
const CODE_RE = /```[\s\S]*?(?:```|$)|`[^`\n]*`/gu;

const outsideCode = (text: string): string[] => text.split(CODE_RE);

// Retrieval numbers become reading order: the first source the answer cites is
// 1, the next 2, and the sources it never cites follow in their old order.
export const citationNumbering = (
  text: string,
  sources: readonly RetrievedSource[],
): ReadonlyMap<number, number> => {
  const known = new Set(
    sources.map((source) => source.ref).filter((ref) => ref !== undefined),
  );
  const order = new Map<number, number>();
  const assign = (ref: number) => {
    if (known.has(ref) && !order.has(ref)) {
      order.set(ref, order.size + 1);
    }
  };

  const cited = outsideCode(text)
    .flatMap((chunk) => chunk.match(CITATION_RUN_RE) ?? [])
    .flatMap((run) => run.match(CITATION_NUMBER_RE) ?? []);

  for (const ref of cited) {
    assign(Number(ref));
  }

  const uncited = [...known].toSorted((a, b) => a - b);

  for (const ref of uncited) {
    assign(ref);
  }

  return order;
};

export const renumberSources = (
  sources: readonly RetrievedSource[],
  numbering: ReadonlyMap<number, number>,
): RetrievedSource[] =>
  sources.map((source) =>
    source.ref === undefined
      ? source
      : { ...source, ref: numbering.get(source.ref) ?? source.ref },
  );

// The copied answer cites the same numbers the reader saw; numbers no source
// carries are dropped, as the chips drop them.
export const renumberText = (
  text: string,
  numbering: ReadonlyMap<number, number>,
): string => {
  if (numbering.size === 0) {
    return text;
  }

  const code = text.match(CODE_RE) ?? [];

  return outsideCode(text)
    .map((chunk, index) => {
      const renumbered = chunk.replaceAll(CITATION_RUN_RE, (run) =>
        (run.match(CITATION_NUMBER_RE) ?? [])
          .map((ref) => numbering.get(Number(ref)))
          .filter((ref) => ref !== undefined)
          .map((ref) => `[${ref}]`)
          .join(''),
      );

      return `${renumbered}${code[index] ?? ''}`;
    })
    .join('');
};

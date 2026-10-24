import type { RetrievedSource, RetrievedSourceLink } from '@/lib/api-types';

export type SourceGroup = {
  key: string;
  sources: readonly RetrievedSource[];
};

// Several searches often return chunks of the same document; they share one
// card. FAQs are separate answers, so each keeps its own.
export const groupSources = (
  sources: readonly RetrievedSource[],
): SourceGroup[] => {
  const groups = new Map<string, RetrievedSource[]>();

  for (const source of sources) {
    const key =
      source.kind === 'faq'
        ? `faq:${source.id}`
        : `${source.kind}:${source.title}`;
    const group = groups.get(key);

    if (group === undefined) {
      groups.set(key, [source]);
    } else {
      group.push(source);
    }
  }

  return [...groups].map(([key, grouped]) => ({ key, sources: grouped }));
};

export const groupSections = (
  sources: readonly RetrievedSource[],
): string[] => [
  ...new Set(
    sources
      .map((source) => source.section?.trim() ?? '')
      .filter((section) => section !== ''),
  ),
];

export const groupLinks = (
  sources: readonly RetrievedSource[],
): RetrievedSourceLink[] => {
  const links = new Map<string, RetrievedSourceLink>();

  for (const source of sources) {
    const sourceLinks = source.links ?? [];

    for (const link of sourceLinks) {
      if (!links.has(link.url)) {
        links.set(link.url, link);
      }
    }
  }

  const authorityUrl = sources.find(
    (source) => source.authorityUrl !== undefined,
  )?.authorityUrl;

  // The API usually lists the authority URL among the links already.
  return authorityUrl !== undefined && links.size === 0
    ? [{ label: sources[0]?.title ?? authorityUrl, url: authorityUrl }]
    : links.values().toArray();
};

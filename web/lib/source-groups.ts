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

export const sourceGroupCount = (sources: readonly RetrievedSource[]): number =>
  groupSources(sources).length;

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

const DAY_RE = /^(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})$/u;
const MONTH_RE = /^(?<year>\d{4})-(?<month>\d{2})$/u;
const YEAR_RE = /^\d{4}$/u;

// Corpus dates are ISO with day, month or year precision, or "unresolved".
export const formatDocumentDate = (
  value: string | undefined,
): null | string => {
  if (value === undefined) {
    return null;
  }

  const day = DAY_RE.exec(value)?.groups;

  if (day !== undefined) {
    return `${day['day']}.${day['month']}.${day['year']}`;
  }

  const month = MONTH_RE.exec(value)?.groups;

  if (month !== undefined) {
    return `${month['month']}.${month['year']}`;
  }

  return YEAR_RE.test(value) ? value : null;
};

// Corpus statuses that mean a newer document probably replaced this one.
const OUTDATED_STATUSES = new Set([
  'likely_superseded',
  'stale_review_required',
]);

export const isOutdatedSource = (source: RetrievedSource): boolean =>
  source.currentStatus !== undefined &&
  OUTDATED_STATUSES.has(source.currentStatus);

const PDF_RE = /\.pdf$/iu;
const WWW_RE = /^w{3}\./u;

const GITHUB_HOSTS = new Set(['github.com', 'raw.githubusercontent.com']);

const hostName = (url: URL): string =>
  GITHUB_HOSTS.has(url.hostname) ? 'GitHub' : url.hostname.replace(WWW_RE, '');

// Visible text for a source link: a document file is named by its type and host,
// since documents usually link both the official copy and a GitHub mirror; a page
// by its label when that says more than the card title, else by its host.
export const sourceLinkText = (
  link: RetrievedSourceLink,
  title: string,
): string => {
  let url: URL;

  try {
    url = new URL(link.url);
  } catch {
    return link.label;
  }

  if (PDF_RE.test(url.pathname)) {
    return `PDF · ${hostName(url)}`;
  }

  const labelIsUseful = link.label.trim() !== '' && link.label !== title;

  return labelIsUseful ? link.label : hostName(url);
};

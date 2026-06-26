const FALLBACK_TITLE = 'Нов разговор';
const TITLE_MAX = 60;

export const deriveTitle = (firstUserText: string): string => {
  const firstLine = firstUserText.split('\n', 1)[0]?.trim() ?? '';

  if (firstLine.length === 0) {
    return FALLBACK_TITLE;
  }

  if (firstLine.length <= TITLE_MAX) {
    return firstLine;
  }

  return `${firstLine.slice(0, TITLE_MAX - 1)}…`;
};

const FALLBACK_TITLE = 'Нов разговор';
const TITLE_MAX = 60;

export const deriveTitle = (firstUserText: string): string => {
  const newlineIndex = firstUserText.indexOf('\n');
  const firstLine = (
    newlineIndex === -1 ? firstUserText : firstUserText.slice(0, newlineIndex)
  ).trim();

  if (firstLine.length === 0) {
    return FALLBACK_TITLE;
  }

  return firstLine.length <= TITLE_MAX
    ? firstLine
    : `${firstLine.slice(0, TITLE_MAX - 1)}…`;
};

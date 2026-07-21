import type { ConversationRow } from '@/lib/conversation-types';

export const DESKTOP_SIDEBAR_QUERY = '(min-width: 1024px)';
const COMPACT_SIDEBAR_QUERY = '(max-width: 1023px)';

type ConversationFilter = {
  readonly filtered: ConversationRow[];
  readonly term: string;
};

export const closeSidebarOnMobile = (onClose: () => void): void => {
  if (
    typeof matchMedia === 'function' &&
    matchMedia(COMPACT_SIDEBAR_QUERY).matches
  ) {
    onClose();
  }
};

export const getSidebarWidthClass = (
  open: boolean,
  synced: boolean,
): string => {
  if (!synced) {
    return 'lg:w-64';
  }

  return open ? 'lg:w-64' : 'lg:w-0';
};

export const getConversationFilter = (
  conversations: ConversationRow[],
  query: string,
): ConversationFilter => {
  const term = query.trim().toLowerCase();

  return {
    filtered: term
      ? conversations.filter((conversation) =>
          conversation.title.toLowerCase().includes(term),
        )
      : conversations,
    term,
  };
};

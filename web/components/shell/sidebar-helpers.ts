import type { ConversationRow } from '@/lib/conversation-types';

type ConversationFilter = {
  readonly filtered: ConversationRow[];
  readonly term: string;
};

export const closeSidebarOnMobile = (onClose: () => void): void => {
  if (
    typeof matchMedia === 'function' &&
    matchMedia('(max-width: 767px)').matches
  ) {
    onClose();
  }
};

export const getSidebarWidthClass = (
  open: boolean,
  synced: boolean,
): string => {
  if (!synced) {
    return 'md:w-64';
  }

  return open ? 'md:w-64' : 'md:w-0';
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

import { Plus } from 'lucide-react';

import type { ConversationRow } from '@/lib/db';

import { ConversationList } from '@/components/shell/conversation-list';
import { t } from '@/lib/i18n';

export type SidebarProps = {
  activeId: null | string;
  conversations: ConversationRow[];
  onDelete: (id: string) => void;
  onNewChat: () => void;
  onRename: (id: string, title: string) => void;
  onSelect: (id: string) => void;
  open: boolean;
};

export const Sidebar = ({
  activeId,
  conversations,
  onDelete,
  onNewChat,
  onRename,
  onSelect,
  open,
}: SidebarProps) => {
  if (!open) {
    return (
      <aside
        aria-label={t('sidebar.label')}
        className="w-0 overflow-hidden"
        data-collapsed="true"
      />
    );
  }

  return (
    <aside
      aria-label={t('sidebar.label')}
      className="flex w-64 shrink-0 flex-col gap-3 border-r border-border bg-muted/30 p-3"
    >
      <button
        className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium hover:bg-muted"
        onClick={onNewChat}
        type="button"
      >
        <Plus
          aria-hidden="true"
          className="size-4"
        />
        {t('sidebar.new')}
      </button>
      <nav className="flex-1 overflow-y-auto">
        <ConversationList
          activeId={activeId}
          conversations={conversations}
          onDelete={onDelete}
          onRename={onRename}
          onSelect={onSelect}
        />
      </nav>
    </aside>
  );
};

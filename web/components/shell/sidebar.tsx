import { Plus } from 'lucide-react';

import type { ConversationRow } from '@/lib/db';

import { ConversationList } from '@/components/shell/conversation-list';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

export type SidebarProps = {
  activeId: null | string;
  conversations: ConversationRow[];
  onClose: () => void;
  onDelete: (id: string) => void;
  onNewChat: () => void;
  onRename: (id: string, title: string) => void;
  onSelect: (id: string) => void;
  open: boolean;
};

const closeIfMobile = (onClose: () => void) => {
  if (
    typeof matchMedia === 'function' &&
    matchMedia('(max-width: 767px)').matches
  ) {
    onClose();
  }
};

export const Sidebar = ({
  activeId,
  conversations,
  onClose,
  onDelete,
  onNewChat,
  onRename,
  onSelect,
  open,
}: SidebarProps) => {
  const handleSelect = (id: string) => {
    onSelect(id);
    closeIfMobile(onClose);
  };

  const handleNewChat = () => {
    onNewChat();
    closeIfMobile(onClose);
  };

  return (
    <>
      {open ? (
        <button
          aria-label={t('header.toggleSidebar')}
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
          onClick={onClose}
          type="button"
        />
      ) : null}
      <aside
        aria-hidden={!open}
        aria-label={t('sidebar.label')}
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-64 shrink-0 overflow-hidden border-r border-border/60 bg-background transition-transform duration-300 ease-in-out md:static md:z-auto md:bg-muted/30 md:transition-[width]',
          open
            ? 'translate-x-0 md:w-64'
            : '-translate-x-full md:w-0 md:translate-x-0',
        )}
        data-collapsed={!open}
      >
        <div className="flex h-full w-64 flex-col gap-3 p-3">
          <button
            className="group inline-flex items-center gap-2 rounded-xl border border-border bg-card px-3 py-2.5 text-sm font-medium shadow-sm transition-all duration-200 hover:border-primary/40 hover:bg-muted hover:shadow active:scale-[0.99]"
            onClick={handleNewChat}
            type="button"
          >
            <Plus
              aria-hidden="true"
              className="size-4 transition-transform duration-200 group-hover:rotate-90"
            />
            {t('sidebar.new')}
          </button>
          <nav className="flex min-h-0 flex-1 flex-col overflow-y-auto">
            <p className="px-2 pb-1.5 pt-1 text-xs font-medium uppercase tracking-wide text-muted-foreground/70">
              {t('sidebar.history')}
            </p>
            <ConversationList
              activeId={activeId}
              conversations={conversations}
              onDelete={onDelete}
              onRename={onRename}
              onSelect={handleSelect}
            />
          </nav>
        </div>
      </aside>
    </>
  );
};

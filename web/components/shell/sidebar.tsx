import { Plus, Search, Trash2, X } from 'lucide-react';
import { useState } from 'react';

import type { ConversationRow } from '@/lib/db';

import { ConversationList } from '@/components/shell/conversation-list';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group';
import { t } from '@/lib/i18n';
import { cn } from '@/lib/utils';

export type SidebarProps = {
  activeId: null | string;
  conversations: ConversationRow[];
  onClearAll: () => void;
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
  onClearAll,
  onClose,
  onDelete,
  onNewChat,
  onRename,
  onSelect,
  open,
}: SidebarProps) => {
  const [confirmingClearAll, setConfirmingClearAll] = useState(false);
  const [query, setQuery] = useState('');

  const handleSelect = (id: string) => {
    onSelect(id);
    closeIfMobile(onClose);
  };

  const handleNewChat = () => {
    onNewChat();
    closeIfMobile(onClose);
  };

  const term = query.trim().toLowerCase();
  const filtered = term
    ? conversations.filter((c) => c.title.toLowerCase().includes(term))
    : conversations;

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
          {conversations.length > 0 ? (
            <InputGroup>
              <InputGroupAddon>
                <Search aria-hidden="true" />
              </InputGroupAddon>
              <InputGroupInput
                aria-label={t('sidebar.search')}
                className="[&::-webkit-search-cancel-button]:appearance-none"
                data-testid="conversation-search"
                onChange={(e) => {
                  setQuery(e.target.value);
                }}
                placeholder={t('sidebar.search')}
                type="search"
                value={query}
              />
              {query ? (
                <InputGroupAddon align="inline-end">
                  <InputGroupButton
                    aria-label={t('sidebar.clearSearch')}
                    onClick={() => {
                      setQuery('');
                    }}
                    size="icon-xs"
                  >
                    <X aria-hidden="true" />
                  </InputGroupButton>
                </InputGroupAddon>
              ) : null}
            </InputGroup>
          ) : null}
          <nav className="flex min-h-0 flex-1 flex-col overflow-y-auto">
            <p className="px-2 pb-1.5 pt-1 text-xs font-medium uppercase tracking-wide text-muted-foreground/70">
              {t('sidebar.history')}
            </p>
            {term && filtered.length === 0 ? (
              <p
                className="px-2 py-1.5 text-sm text-muted-foreground"
                data-testid="no-results"
              >
                {t('sidebar.noResults')}
              </p>
            ) : (
              <ConversationList
                activeId={activeId}
                conversations={filtered}
                onDelete={onDelete}
                onRename={onRename}
                onSelect={handleSelect}
              />
            )}
          </nav>
          {conversations.length > 0 ? (
            <button
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-border/60 px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:border-destructive/40 hover:bg-destructive/10 hover:text-destructive"
              data-testid="delete-all"
              onClick={() => {
                setConfirmingClearAll(true);
              }}
              type="button"
            >
              <Trash2
                aria-hidden="true"
                className="size-4"
              />
              {t('sidebar.deleteAll')}
            </button>
          ) : null}
        </div>
      </aside>
      <ConfirmDialog
        confirmLabel={t('conversation.delete')}
        description={t('conversation.deleteAllDescription')}
        destructive
        onConfirm={onClearAll}
        onOpenChange={setConfirmingClearAll}
        open={confirmingClearAll}
        title={t('conversation.deleteAllTitle')}
      />
    </>
  );
};

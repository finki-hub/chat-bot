import { Plus, Search, X } from 'lucide-react';
import { type ReactNode, useEffect, useRef, useState } from 'react';

import type { MaybeAsyncAction } from '@/lib/action-result';
import type {
  ChatSearchResult,
  ConversationRow,
} from '@/lib/conversation-types';

import { ConversationList } from '@/components/shell/conversation-list';
import { ConversationSearchResults } from '@/components/shell/conversation-search-results';
import {
  closeSidebarOnMobile,
  getConversationFilter,
  getSidebarWidthClass,
} from '@/components/shell/sidebar-helpers';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group';
import { Spinner } from '@/components/ui/spinner';
import { t } from '@/lib/i18n';
import { searchChatConversations } from '@/lib/transport';
import { cn } from '@/lib/utils';

export type SidebarProps = {
  activeId: null | string;
  conversations: ConversationRow[];
  footer?: ReactNode;
  generatingTitleId?: null | string;
  listError?: boolean;
  listLoading?: boolean;
  mobile?: boolean;
  onClose: () => void;
  onDelete: MaybeAsyncAction<[id: string]>;
  onGenerateTitle?: (id: string) => void;
  onNewChat: () => void;
  onOpen?: () => void;
  onRename: MaybeAsyncAction<[id: string, title: string]>;
  onRetryList?: () => Promise<void>;
  onSelect: (id: string) => void;
  onSelectMessage?: (
    conversationId: string,
    messageId: null | string,
    conversationTitle: null | string,
  ) => void;
  open: boolean;
  synced?: boolean;
};

/* eslint-disable sonarjs/cognitive-complexity -- responsive drawer, filtering, and confirmation form one navigation state machine */
const mergeLocalTitleMatches = (
  results: readonly ChatSearchResult[],
  localMatches: readonly ConversationRow[],
): ChatSearchResult[] => {
  const matchedIds = new Set(results.map((result) => result.conversationId));
  return [
    ...results,
    ...localMatches
      .filter((conversation) => !matchedIds.has(conversation.id))
      .map((conversation) => ({
        conversationId: conversation.id,
        conversationTitle: conversation.title,
        conversationUpdatedAt: conversation.updatedAt ?? '',
        message: null,
        snippet: null,
      })),
  ];
};

export const Sidebar = ({
  activeId,
  conversations,
  footer,
  generatingTitleId,
  listError = false,
  listLoading = false,
  mobile = false,
  onClose,
  onDelete,
  onGenerateTitle,
  onNewChat,
  onOpen,
  onRename,
  onRetryList,
  onSelect,
  onSelectMessage,
  open,
  synced = true,
}: SidebarProps) => {
  const [query, setQuery] = useState('');
  const [searchError, setSearchError] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchNonce, setSearchNonce] = useState(0);
  const [searchResults, setSearchResults] = useState<ChatSearchResult[]>([]);
  const mobileTriggerRef = useRef<HTMLElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const trimmedQuery = query.trim();
  const serverSearch = trimmedQuery.length >= 2;

  useEffect(() => {
    if (!mobile) {
      mobileTriggerRef.current = null;
      return;
    }
    if (open) {
      mobileTriggerRef.current =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
      return;
    }
    mobileTriggerRef.current?.focus();
    mobileTriggerRef.current = null;
  }, [mobile, open]);

  useEffect(() => {
    let focusTimer: null | ReturnType<typeof setTimeout> = null;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!(
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === 'k'
      )) {
        return;
      }

      event.preventDefault();
      if (!open) {
        onOpen?.();
      }
      focusTimer = setTimeout(() => {
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      }, 0);
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (focusTimer !== null) {
        clearTimeout(focusTimer);
      }
    };
  }, [onOpen, open]);

  useEffect(() => {
    if (!serverSearch) {
      setSearchError(false);
      setSearchLoading(false);
      setSearchResults([]);
      return () => {};
    }

    const controller = new AbortController();
    setSearchError(false);
    setSearchLoading(true);
    setSearchResults([]);

    const timer = setTimeout(() => {
      void (async () => {
        try {
          const results = await searchChatConversations(
            trimmedQuery,
            controller.signal,
          );
          if (controller.signal.aborted) {
            return;
          }
          setSearchResults(results);
          setSearchLoading(false);
        } catch {
          if (controller.signal.aborted) {
            return;
          }
          setSearchError(true);
          setSearchLoading(false);
        }
      })();
    }, 250);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [searchNonce, serverSearch, trimmedQuery]);

  const handleSearchSelect = (
    conversationId: string,
    messageId: null | string,
    conversationTitle: null | string,
  ) => {
    onSelectMessage?.(conversationId, messageId, conversationTitle);
    closeSidebarOnMobile(onClose);
  };

  const handleSelect = (id: string) => {
    onSelect(id);
    closeSidebarOnMobile(onClose);
  };

  const handleNewChat = () => {
    onNewChat();
    closeSidebarOnMobile(onClose);
  };

  const { filtered, term } = getConversationFilter(conversations, query);
  const responsiveStateClass = getSidebarWidthClass(open, synced);
  const visibleSearchResults = mergeLocalTitleMatches(searchResults, filtered);

  const renderLocalConversations = (): ReactNode =>
    term && filtered.length === 0 ? (
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
        generatingTitleId={generatingTitleId}
        onDelete={onDelete}
        onGenerateTitle={onGenerateTitle}
        onRename={onRename}
        onSelect={handleSelect}
      />
    );

  // eslint-disable-next-line sonarjs/function-return-type -- JSX branches render different elements
  const renderConversationBody = (): ReactNode => {
    if (serverSearch) {
      if (searchLoading) {
        return (
          <>
            <output
              className="flex items-center gap-2 px-2 py-1.5 text-sm text-muted-foreground"
              data-testid="search-loading"
            >
              <Spinner className="size-4" />
              {t('search.loading')}
            </output>
            {renderLocalConversations()}
          </>
        );
      }
      if (searchError) {
        return (
          <>
            <div
              className="rounded-lg border border-destructive/40 bg-destructive/10 p-2 text-sm"
              role="alert"
            >
              <p className="text-destructive">{t('search.error')}</p>
              <button
                className="mt-2 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:min-h-11"
                onClick={() => {
                  setSearchNonce((nonce) => nonce + 1);
                }}
                type="button"
              >
                {t('error.retry')}
              </button>
            </div>
            {renderLocalConversations()}
          </>
        );
      }
      if (visibleSearchResults.length === 0) {
        return (
          <p
            className="px-2 py-1.5 text-sm text-muted-foreground"
            data-testid="no-results"
          >
            {t('sidebar.noResults')}
          </p>
        );
      }
      return (
        <ConversationSearchResults
          onSelect={handleSearchSelect}
          results={visibleSearchResults}
        />
      );
    }

    return renderLocalConversations();
  };

  const sidebarContent = (
    // Fixed width on desktop so the content doesn't reflow as the panel slides.
    <div
      className={`${mobile ? 'w-full min-w-0' : 'w-64'} flex h-full flex-col gap-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] pl-[max(0.75rem,env(safe-area-inset-left))] pr-[max(0.75rem,env(safe-area-inset-right))] pt-[max(0.75rem,env(safe-area-inset-top))]`}
    >
      <button
        className="group inline-flex min-h-11 items-center gap-2 rounded-xl border border-border bg-card px-3 py-2.5 text-sm font-medium shadow-sm transition-[background-color,border-color,box-shadow,transform] duration-200 hover:border-primary/40 hover:bg-muted hover:shadow active:scale-[0.99]"
        onClick={handleNewChat}
        type="button"
      >
        <Plus
          aria-hidden="true"
          className="size-4 transition-transform duration-200 group-hover:rotate-90"
        />
        {t('sidebar.new')}
      </button>
      {listError ? (
        <div
          className="rounded-lg border border-destructive/40 bg-destructive/10 p-2 text-sm"
          role="alert"
        >
          <p className="text-destructive">{t('sidebar.loadError')}</p>
          {onRetryList ? (
            <button
              className="mt-2 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:min-h-11"
              onClick={() => {
                void onRetryList();
              }}
              type="button"
            >
              {t('error.retry')}
            </button>
          ) : null}
        </div>
      ) : null}
      {listLoading && conversations.length === 0 ? (
        <output className="px-2 py-1.5 text-sm text-muted-foreground">
          {t('sidebar.loading')}
        </output>
      ) : null}
      {conversations.length > 0 ? (
        <InputGroup>
          <InputGroupAddon>
            <Search aria-hidden="true" />
          </InputGroupAddon>
          <InputGroupInput
            aria-label={t('sidebar.search')}
            className="[&::-webkit-search-cancel-button]:appearance-none"
            data-testid="search-conversations"
            onChange={(e) => {
              setQuery(e.target.value);
            }}
            placeholder={t('sidebar.search')}
            ref={searchInputRef}
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
                onMouseDown={(e) => {
                  e.preventDefault();
                }}
                size="icon-xs"
              >
                <X aria-hidden="true" />
              </InputGroupButton>
            </InputGroupAddon>
          ) : null}
        </InputGroup>
      ) : null}
      {/* The date groups head the list, so the section needs no visible title. */}
      <nav
        aria-label={serverSearch ? t('search.results') : t('sidebar.history')}
        className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain pt-1"
      >
        {renderConversationBody()}
      </nav>
      {footer}
    </div>
  );

  return mobile ? (
    <Dialog
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          onClose();
        }
      }}
      open={open}
    >
      <DialogContent
        aria-describedby={undefined}
        className="left-0 top-0 h-dvh max-h-none w-[min(85vw,20rem)] max-w-none translate-x-0 translate-y-0 gap-0 overflow-hidden rounded-none border-y-0 border-l-0 p-0"
        showCloseButton={false}
      >
        <DialogTitle className="sr-only">{t('sidebar.label')}</DialogTitle>
        {sidebarContent}
      </DialogContent>
    </Dialog>
  ) : (
    <aside
      aria-hidden={synced ? !open : undefined}
      aria-label={t('sidebar.label')}
      className={cn(
        'static z-auto w-64 shrink-0 overflow-hidden border-r border-border/60 bg-muted/30',
        synced
          ? 'transition-[width] duration-300 ease-in-out'
          : 'transition-none',
        responsiveStateClass,
      )}
      data-collapsed={synced ? !open : undefined}
      inert={synced && !open}
    >
      {sidebarContent}
    </aside>
  );
};
/* eslint-enable sonarjs/cognitive-complexity -- restore the rule after the navigation state machine */

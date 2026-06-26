import { Pencil, Trash2 } from 'lucide-react';

import type { ConversationRow } from '@/lib/db';

import { t } from '@/lib/i18n';

export type ConversationListProps = {
  activeId: null | string;
  conversations: ConversationRow[];
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onSelect: (id: string) => void;
};

export const ConversationList = ({
  activeId,
  conversations,
  onDelete,
  onRename,
  onSelect,
}: ConversationListProps) => (
  <ul className="flex flex-col gap-1">
    {conversations.map((c) => (
      <li
        aria-current={c.id === activeId ? 'true' : undefined}
        className={`group flex items-center justify-between gap-1 rounded-lg px-2.5 py-2 text-sm text-foreground/80 transition-colors duration-150 hover:bg-muted/70 ${
          c.id === activeId ? 'bg-muted font-medium text-foreground' : ''
        }`}
        data-testid={`conversation-${c.id}`}
        key={c.id}
      >
        <button
          className="flex-1 truncate text-left"
          onClick={() => {
            onSelect(c.id);
          }}
          type="button"
        >
          {c.title}
        </button>
        <span className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
          <button
            aria-label={t('conversation.rename')}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
            onClick={() => {
              // eslint-disable-next-line no-alert -- lightweight rename UX; spec §9 uses the native prompt
              const next = prompt(t('conversation.renamePrompt'), c.title);
              if (next?.trim()) {
                onRename(c.id, next.trim());
              }
            }}
            type="button"
          >
            <Pencil
              aria-hidden="true"
              className="size-3.5"
            />
          </button>
          <button
            aria-label={t('conversation.delete')}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-background hover:text-destructive"
            onClick={() => {
              onDelete(c.id);
            }}
            type="button"
          >
            <Trash2
              aria-hidden="true"
              className="size-3.5"
            />
          </button>
        </span>
      </li>
    ))}
  </ul>
);

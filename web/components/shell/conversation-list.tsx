import { Pencil, Trash2 } from 'lucide-react';
import { useState } from 'react';

import type { ConversationRow } from '@/lib/db';

import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
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
}: ConversationListProps) => {
  const [renameTarget, setRenameTarget] = useState<ConversationRow | null>(
    null,
  );
  const [renameValue, setRenameValue] = useState('');
  const [pendingDelete, setPendingDelete] = useState<ConversationRow | null>(
    null,
  );

  const openRename = (conversation: ConversationRow) => {
    setRenameTarget(conversation);
    setRenameValue(conversation.title);
  };

  const submitRename = () => {
    const next = renameValue.trim();
    if (renameTarget && next.length > 0) {
      onRename(renameTarget.id, next);
    }
    setRenameTarget(null);
  };

  const confirmDelete = () => {
    if (pendingDelete) {
      onDelete(pendingDelete.id);
    }
    setPendingDelete(null);
  };

  return (
    <>
      <ul className="flex flex-col gap-1">
        {conversations.map((c) => (
          <li
            aria-current={c.id === activeId ? 'true' : undefined}
            className={`group flex items-center justify-between gap-1 rounded-lg px-2.5 py-1.5 text-sm text-foreground/80 transition-colors duration-150 hover:bg-muted/70 ${
              c.id === activeId ? 'bg-muted font-medium text-foreground' : ''
            }`}
            data-testid={`conversation-${c.id}`}
            key={c.id}
          >
            <button
              className="flex-1 truncate rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
              onClick={() => {
                onSelect(c.id);
              }}
              type="button"
            >
              {c.title}
            </button>
            <span
              className="flex items-center gap-1 opacity-100 transition-opacity duration-150 sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100"
              data-testid="row-actions"
            >
              <button
                aria-label={t('conversation.rename')}
                className="rounded-md p-1 text-muted-foreground outline-none transition-colors hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
                onClick={() => {
                  openRename(c);
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
                className="rounded-md p-1 text-muted-foreground outline-none transition-colors hover:bg-background hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring/50"
                onClick={() => {
                  setPendingDelete(c);
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

      <Dialog
        onOpenChange={(open) => {
          if (!open) {
            setRenameTarget(null);
          }
        }}
        open={renameTarget !== null}
      >
        <DialogContent
          aria-describedby={undefined}
          className="sm:max-w-sm"
        >
          <DialogHeader>
            <DialogTitle>{t('conversation.renameTitle')}</DialogTitle>
          </DialogHeader>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submitRename();
            }}
          >
            <Input
              aria-label={t('conversation.renamePrompt')}
              autoComplete="off"
              name="conversation-title"
              onChange={(e) => {
                setRenameValue(e.target.value);
              }}
              value={renameValue}
            />
            <DialogFooter className="mt-4">
              <Button
                onClick={() => {
                  setRenameTarget(null);
                }}
                type="button"
                variant="outline"
              >
                {t('common.cancel')}
              </Button>
              <Button
                data-testid="confirm-rename"
                disabled={renameValue.trim().length === 0}
                type="submit"
              >
                {t('common.save')}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        confirmLabel={t('conversation.delete')}
        description={t('conversation.deleteDescription')}
        destructive
        onConfirm={confirmDelete}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDelete(null);
          }
        }}
        open={pendingDelete !== null}
        title={t('conversation.deleteTitle')}
      />
    </>
  );
};

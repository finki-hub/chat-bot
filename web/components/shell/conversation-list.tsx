import { LoaderCircle, Pencil, Trash2, WandSparkles } from 'lucide-react';
import { useState } from 'react';

import type { MaybeAsyncAction } from '@/lib/action-result';
import type { ConversationRow } from '@/lib/conversation-types';

import { ConversationActionTooltip } from '@/components/shell/conversation-action-tooltip';
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
import { Spinner } from '@/components/ui/spinner';
import { fireAndForget } from '@/lib/async';
import { t } from '@/lib/i18n';

export type ConversationListProps = {
  activeId: null | string;
  conversations: ConversationRow[];
  generatingTitleId?: null | string;
  onDelete: MaybeAsyncAction<[id: string]>;
  onGenerateTitle?: (id: string) => void;
  onRename: MaybeAsyncAction<[id: string, title: string]>;
  onSelect: (id: string) => void;
};

export const ConversationList = ({
  activeId,
  conversations,
  generatingTitleId = null,
  onDelete,
  onGenerateTitle,
  onRename,
  onSelect,
}: ConversationListProps) => {
  const [renameTarget, setRenameTarget] = useState<ConversationRow | null>(
    null,
  );
  const [renameValue, setRenameValue] = useState('');
  const [renameFailed, setRenameFailed] = useState(false);
  const [renamePending, setRenamePending] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ConversationRow | null>(
    null,
  );
  const isGeneratingAnyTitle = generatingTitleId !== null;

  const openRename = (conversation: ConversationRow) => {
    setRenameFailed(false);
    setRenameTarget(conversation);
    setRenameValue(conversation.title);
  };

  const submitRename = async (): Promise<void> => {
    const next = renameValue.trim();
    if (renamePending || renameTarget === null || next.length === 0) {
      return;
    }
    setRenameFailed(false);
    setRenamePending(true);
    try {
      const renamed = await onRename(renameTarget.id, next);
      if (renamed === false) {
        setRenameFailed(true);
        return;
      }
      setRenameTarget(null);
    } finally {
      setRenamePending(false);
    }
  };

  const confirmDelete = () =>
    pendingDelete === null ? undefined : onDelete(pendingDelete.id);

  return (
    <>
      <ul className="flex flex-col gap-1">
        {conversations.map((c) => {
          const isGeneratingTitle = generatingTitleId === c.id;

          return (
            <li
              aria-current={c.id === activeId ? 'true' : undefined}
              className={`group flex items-center justify-between gap-1 rounded-lg px-2.5 py-1.5 text-sm text-foreground/80 transition-colors duration-150 hover:bg-muted/70 ${
                c.id === activeId ? 'bg-muted font-medium text-foreground' : ''
              }`}
              data-testid={`conversation-${c.id}`}
              key={c.id}
            >
              <button
                className="min-h-11 flex-1 truncate rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50 sm:pointer-fine:min-h-0"
                onClick={() => {
                  onSelect(c.id);
                }}
                type="button"
              >
                {c.title}
              </button>
              <span
                className="flex items-center gap-1 opacity-100 transition-opacity duration-150 sm:pointer-fine:opacity-0 sm:pointer-fine:group-focus-within:opacity-100 sm:pointer-fine:group-hover:opacity-100"
                data-testid="row-actions"
              >
                {onGenerateTitle ? (
                  <ConversationActionTooltip
                    disabled={isGeneratingAnyTitle}
                    label={t('conversation.generateTitle')}
                  >
                    <button
                      aria-busy={isGeneratingTitle || undefined}
                      aria-label={t('conversation.generateTitle')}
                      className="inline-flex size-11 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-primary focus-visible:ring-2 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-60 sm:pointer-fine:size-6"
                      disabled={isGeneratingAnyTitle}
                      onClick={() => {
                        onGenerateTitle(c.id);
                      }}
                      type="button"
                    >
                      {isGeneratingTitle ? (
                        <LoaderCircle
                          aria-hidden="true"
                          className="size-3.5 animate-spin"
                        />
                      ) : (
                        <WandSparkles
                          aria-hidden="true"
                          className="size-3.5"
                        />
                      )}
                    </button>
                  </ConversationActionTooltip>
                ) : null}
                <ConversationActionTooltip label={t('conversation.rename')}>
                  <button
                    aria-label={t('conversation.rename')}
                    className="inline-flex size-11 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 sm:pointer-fine:size-6"
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
                </ConversationActionTooltip>
                <ConversationActionTooltip label={t('conversation.delete')}>
                  <button
                    aria-label={t('conversation.delete')}
                    className="inline-flex size-11 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring/50 sm:pointer-fine:size-6"
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
                </ConversationActionTooltip>
              </span>
            </li>
          );
        })}
      </ul>

      <Dialog
        onOpenChange={(open) => {
          if (!open && !renamePending) {
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
              fireAndForget(submitRename());
            }}
          >
            <Input
              aria-label={t('conversation.renamePrompt')}
              autoComplete="off"
              disabled={renamePending}
              name="conversation-title"
              onChange={(e) => {
                setRenameFailed(false);
                setRenameValue(e.target.value);
              }}
              value={renameValue}
            />
            {renameFailed ? (
              <p
                className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                role="alert"
              >
                {t('conversation.renameError')}
              </p>
            ) : null}
            <DialogFooter className="mt-4">
              <Button
                disabled={renamePending}
                onClick={() => {
                  setRenameFailed(false);
                  setRenameTarget(null);
                }}
                type="button"
                variant="outline"
              >
                {t('common.cancel')}
              </Button>
              <Button
                aria-busy={renamePending || undefined}
                data-testid="confirm-rename"
                disabled={renamePending || renameValue.trim().length === 0}
                type="submit"
              >
                {renamePending ? <Spinner aria-hidden="true" /> : null}
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
        errorMessage={t('conversation.deleteError')}
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

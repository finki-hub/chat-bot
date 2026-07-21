import {
  Ellipsis,
  LoaderCircle,
  Pencil,
  Trash2,
  WandSparkles,
} from 'lucide-react';

import type { ConversationRow } from '@/lib/conversation-types';

import { ConversationActionTooltip } from '@/components/shell/conversation-action-tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { t } from '@/lib/i18n';

type ConversationRowActionsProps = {
  readonly conversation: ConversationRow;
  readonly generatingTitleId: null | string;
  readonly onDeleteAction: (conversation: ConversationRow) => void;
  readonly onGenerateTitle?: (id: string) => void;
  readonly onRenameAction: (conversation: ConversationRow) => void;
};

export const ConversationRowActions = ({
  conversation,
  generatingTitleId,
  onDeleteAction,
  onGenerateTitle,
  onRenameAction,
}: ConversationRowActionsProps) => {
  const isGeneratingAnyTitle = generatingTitleId !== null;
  const isGeneratingTitle = generatingTitleId === conversation.id;
  const actionsLabel = `${t('conversation.actions')}: ${conversation.title}`;

  return (
    <>
      <span
        className="hidden items-center gap-1 opacity-0 transition-opacity duration-150 pointer-fine:flex pointer-fine:group-focus-within:opacity-100 pointer-fine:group-hover:opacity-100"
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
              className="inline-flex size-6 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-primary focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-60"
              disabled={isGeneratingAnyTitle}
              onClick={() => {
                onGenerateTitle(conversation.id);
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
            className="inline-flex size-6 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
            onClick={() => {
              onRenameAction(conversation);
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
            className="inline-flex size-6 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring"
            onClick={() => {
              onDeleteAction(conversation);
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

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            aria-label={actionsLabel}
            className="inline-flex size-12 shrink-0 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring pointer-fine:hidden"
            type="button"
          >
            <Ellipsis
              aria-hidden="true"
              className="size-5"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="min-w-48"
          collisionPadding={12}
        >
          <DropdownMenuGroup>
            {onGenerateTitle ? (
              <DropdownMenuItem
                aria-busy={isGeneratingTitle || undefined}
                className="min-h-11 pointer-fine:min-h-8"
                disabled={isGeneratingAnyTitle}
                onSelect={() => {
                  onGenerateTitle(conversation.id);
                }}
              >
                {isGeneratingTitle ? (
                  <LoaderCircle
                    aria-hidden="true"
                    className="animate-spin"
                  />
                ) : (
                  <WandSparkles aria-hidden="true" />
                )}
                {t('conversation.generateTitle')}
              </DropdownMenuItem>
            ) : null}
            <DropdownMenuItem
              className="min-h-11 pointer-fine:min-h-8"
              onSelect={() => {
                onRenameAction(conversation);
              }}
            >
              <Pencil aria-hidden="true" />
              {t('conversation.rename')}
            </DropdownMenuItem>
            <DropdownMenuItem
              className="min-h-11 pointer-fine:min-h-8"
              onSelect={() => {
                onDeleteAction(conversation);
              }}
              variant="destructive"
            >
              <Trash2 aria-hidden="true" />
              {t('conversation.delete')}
            </DropdownMenuItem>
          </DropdownMenuGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
};

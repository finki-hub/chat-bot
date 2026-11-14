import {
  Ellipsis,
  LoaderCircle,
  Pencil,
  Trash2,
  WandSparkles,
} from 'lucide-react';

import type { ConversationRow } from '@/lib/conversation-types';

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

  // One control per row, so a long history stays quick to tab through. With a
  // mouse it appears on hover or focus; a generating title keeps its spinner shown.
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          aria-busy={isGeneratingTitle || undefined}
          aria-label={actionsLabel}
          className={`inline-flex size-12 shrink-0 items-center justify-center rounded-md text-muted-foreground outline-none transition-[opacity,color] hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring pointer-fine:size-7 pointer-fine:focus-visible:opacity-100 pointer-fine:data-[state=open]:opacity-100 pointer-fine:group-focus-within:opacity-100 pointer-fine:group-hover:opacity-100 ${
            isGeneratingTitle ? '' : 'pointer-fine:opacity-0'
          }`}
          data-testid="row-actions"
          type="button"
        >
          {isGeneratingTitle ? (
            <LoaderCircle
              aria-hidden="true"
              className="size-4 animate-spin"
            />
          ) : (
            <Ellipsis
              aria-hidden="true"
              className="size-5 pointer-fine:size-4"
            />
          )}
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
  );
};

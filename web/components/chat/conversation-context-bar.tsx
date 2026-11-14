import type { ConversationSharing } from '@/components/chat/use-conversation-sharing';

import { ShareConversationButton } from '@/components/chat/share-conversation-button';
import { t } from '@/lib/i18n';

type ConversationContextBarProps = {
  readonly conversationId: string;
  readonly sharing: ConversationSharing;
  readonly title: string;
};

export const ConversationContextBar = ({
  conversationId,
  sharing,
  title,
}: ConversationContextBarProps) => (
  <section
    aria-label={t('conversation.contextLabel')}
    // On phones the app header carries the title and share control instead.
    className="flex min-h-14 shrink-0 max-sm:hidden items-center gap-3 border-b border-border/60 bg-background/95 px-3 sm:min-h-12 sm:px-4 [@media(max-height:500px)]:min-h-11"
    data-testid="chat-context-bar"
  >
    <h2
      className="min-w-0 flex-1 truncate text-sm font-semibold tracking-tight text-foreground"
      title={title}
    >
      {title}
    </h2>
    <ShareConversationButton
      conversationId={conversationId}
      sharing={sharing}
    />
  </section>
);

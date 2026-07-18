import { ShareConversationButton } from '@/components/chat/share-conversation-button';
import { t } from '@/lib/i18n';

type ConversationContextBarProps = {
  readonly conversationId: string;
  readonly title: string;
};

export const ConversationContextBar = ({
  conversationId,
  title,
}: ConversationContextBarProps) => (
  <section
    aria-label={t('conversation.contextLabel')}
    className="flex min-h-14 shrink-0 items-center gap-3 border-b border-border/60 bg-background/95 px-3 sm:min-h-12 sm:px-4"
    data-testid="conversation-context-bar"
  >
    <h2
      className="min-w-0 flex-1 truncate text-sm font-semibold tracking-tight text-foreground"
      title={title}
    >
      {title}
    </h2>
    <ShareConversationButton conversationId={conversationId} />
  </section>
);

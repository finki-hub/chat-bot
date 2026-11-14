import type { ReactNode } from 'react';

import type { ErrorNotice, MyUIMessage, StatusPart } from '@/lib/api-types';

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent } from '@/components/ai-elements/message';
import { ActivityLine } from '@/components/chat/activity-line';
import { AssistantMessage, MessageError } from '@/components/chat/message';
import { UserMessage } from '@/components/chat/user-message';
import { t } from '@/lib/i18n';
import { hasReasoning, hasText, hasToolCalls } from '@/lib/message-parts';

export type ThreadProps = {
  activeError?: ErrorNotice;
  activeStatus?: StatusPart;
  conversationId?: null | string;
  messages: MyUIMessage[];
  // Edit-and-resend for sent questions; offered only while nothing streams.
  onEditMessage?: (messageId: string, text: string) => void;
  onManageCredentials?: (returnFocusTarget: HTMLElement) => void;
  onPickSuggestion?: (text: string) => unknown;
  onRetry?: () => void;
  renderActions?: (message: MyUIMessage) => ReactNode;
  retryPending?: boolean;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
};

const SUGGESTIONS = [
  'Колку кредити ми требаат за\u{A0}да\u{A0}се\u{A0}запишам во наредна година?',
  'Колку изнесува школарината за\u{A0}една\u{A0}учебна\u{A0}година?',
  'Колку пати можам да полагам еден испит?',
  'Кои се условите за дипломирање?',
] as const;

const NON_BREAKING_SPACE = '\u{A0}';

const CHIP =
  'group rounded-xl border border-border bg-card/60 px-4 py-3 text-left text-pretty text-sm text-foreground/90 shadow-sm transition-[background-color,border-color,box-shadow,transform] duration-200 hover:-translate-y-0.5 hover:border-primary/40 hover:bg-card hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';
const MESSAGE_ENTRANCE =
  'motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500';

// Appears as one short fade, suggestions included, so a fresh load is never
// briefly blank or half-built.
const Welcome = ({ onPick }: { onPick?: (text: string) => void }) => (
  <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center gap-6 py-12 text-center motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200 [@media(max-height:500px)]:justify-start [@media(max-height:500px)]:gap-3 [@media(max-height:500px)]:py-3">
    <img
      alt="ФИНКИ Хаб"
      className="h-16 w-16 object-contain drop-shadow-sm [@media(max-height:500px)]:h-12 [@media(max-height:500px)]:w-12"
      height={64}
      src="/logo.png"
      width={64}
    />
    <div className="space-y-2">
      <h2 className="text-2xl font-bold tracking-tight">
        {t('thread.emptyTitle')}
      </h2>
      <p className="text-muted-foreground">{t('thread.emptyDescription')}</p>
    </div>
    {onPick ? (
      <div className="grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            aria-label={s.replaceAll(NON_BREAKING_SPACE, ' ')}
            className={CHIP}
            key={s}
            onClick={() => {
              onPick(s.replaceAll(NON_BREAKING_SPACE, ' '));
            }}
            type="button"
          >
            {s}
          </button>
        ))}
      </div>
    ) : null}
  </div>
);

type ThreadEmptyStateProps = Pick<
  ThreadProps,
  | 'activeError'
  | 'onManageCredentials'
  | 'onPickSuggestion'
  | 'onRetry'
  | 'retryPending'
>;

const ThreadEmptyState = ({
  activeError,
  onManageCredentials,
  onPickSuggestion,
  onRetry,
  retryPending,
}: ThreadEmptyStateProps) =>
  activeError === undefined ? (
    <Welcome onPick={onPickSuggestion} />
  ) : (
    <div className="mx-auto flex w-full max-w-3xl flex-1 items-center px-4 py-8 sm:px-6">
      <div className="w-full">
        <MessageError
          errorPart={activeError}
          onManageCredentials={onManageCredentials}
          onRetry={onRetry}
          retryPending={retryPending}
        />
      </div>
    </div>
  );

export const Thread = ({
  activeError,
  activeStatus,
  conversationId,
  messages,
  onEditMessage,
  onManageCredentials,
  onPickSuggestion,
  onRetry,
  renderActions,
  retryPending,
  status,
}: ThreadProps) => {
  const lastAssistantId = messages.findLast((m) => m.role === 'assistant')?.id;
  const streaming = status === 'streaming' || status === 'submitted';
  const lastMessage = messages.at(-1);
  const editQuestion = streaming ? undefined : onEditMessage;
  const awaitingReply =
    streaming &&
    (lastMessage?.role !== 'assistant' ||
      (activeStatus === undefined &&
        !hasText(lastMessage) &&
        !hasReasoning(lastMessage) &&
        !hasToolCalls(lastMessage)));
  return (
    <Conversation
      className="flex-1"
      initial={messages.length === 0 ? false : 'instant'}
      key={`${messages.at(0)?.id ?? conversationId ?? 'unscoped'}:${messages.length === 0 ? 'welcome' : 'messages'}`}
    >
      <ConversationContent>
        {messages.length === 0 ? (
          <ThreadEmptyState
            activeError={activeError}
            onManageCredentials={onManageCredentials}
            onPickSuggestion={onPickSuggestion}
            onRetry={onRetry}
            retryPending={retryPending}
          />
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
            {messages.map((m) => {
              if (
                activeStatus === undefined &&
                streaming &&
                m.role === 'assistant' &&
                m.id === lastAssistantId &&
                !hasText(m) &&
                !hasReasoning(m) &&
                !hasToolCalls(m)
              ) {
                return null;
              }

              if (m.role === 'user') {
                return (
                  <UserMessage
                    className={
                      streaming && m.id === lastMessage?.id
                        ? MESSAGE_ENTRANCE
                        : undefined
                    }
                    key={m.id}
                    message={m}
                    onEdit={editQuestion}
                  />
                );
              }

              const isLastAssistant = m.id === lastAssistantId;
              const activeAssistant =
                isLastAssistant &&
                lastMessage?.role === 'assistant' &&
                (streaming || activeStatus !== undefined);
              return (
                <div
                  className={activeAssistant ? MESSAGE_ENTRANCE : undefined}
                  key={m.id}
                >
                  <AssistantMessage
                    actions={renderActions ? renderActions(m) : undefined}
                    errorPart={
                      isLastAssistant
                        ? (activeError ?? m.metadata?.error)
                        : m.metadata?.error
                    }
                    message={m}
                    onManageCredentials={
                      isLastAssistant ? onManageCredentials : undefined
                    }
                    onRetry={isLastAssistant ? onRetry : undefined}
                    pending={activeAssistant}
                    retryPending={isLastAssistant ? retryPending : undefined}
                    statusPart={activeAssistant ? activeStatus : undefined}
                  />
                </div>
              );
            })}
            {awaitingReply ? (
              <Message
                className={MESSAGE_ENTRANCE}
                from="assistant"
              >
                <MessageContent>
                  <ActivityLine label={t('activity.processing')} />
                </MessageContent>
              </Message>
            ) : null}
          </div>
        )}
        {activeError && lastAssistantId === undefined && messages.length > 0 ? (
          <div className="mx-auto w-full max-w-3xl">
            <MessageError
              errorPart={activeError}
              onManageCredentials={onManageCredentials}
              onRetry={onRetry}
              retryPending={retryPending}
            />
          </div>
        ) : null}
      </ConversationContent>
      {messages.length === 0 ? null : <ConversationScrollButton />}
    </Conversation>
  );
};

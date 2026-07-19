import type { ReactNode } from 'react';

import type { ErrorNotice, MyUIMessage, StatusPart } from '@/lib/api-types';

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { AssistantMessage, MessageError } from '@/components/chat/message';
import { TypingIndicator } from '@/components/chat/typing-indicator';
import { t } from '@/lib/i18n';
import { hasReasoning, hasText, joinText } from '@/lib/message-parts';

export type ThreadProps = {
  activeError?: ErrorNotice;
  activeStatus?: StatusPart;
  messages: MyUIMessage[];
  onPickSuggestion?: (text: string) => unknown;
  onRetry?: () => void;
  renderActions?: (message: MyUIMessage) => ReactNode;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
};

const SUGGESTIONS = [
  'Колку кредити ми требаат за да се запишам во наредна година?',
  'Колку изнесува школарината за една учебна година?',
  'Колку пати можам да полагам еден испит?',
  'Кои се условите за дипломирање?',
] as const;

const NON_BREAKING_SPACE = ' ';
const STAGGER = ['stagger-1', 'stagger-2', 'stagger-3', 'stagger-4'];

const CHIP =
  'group rounded-xl border border-border bg-card/60 px-4 py-3 text-left text-pretty text-sm text-foreground/90 shadow-sm transition-[background-color,border-color,box-shadow,transform] duration-200 hover:-translate-y-0.5 hover:border-primary/40 hover:bg-card hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:fill-mode-both';

const Welcome = ({ onPick }: { onPick?: (text: string) => void }) => (
  <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center gap-6 py-12 text-center motion-safe:animate-in motion-safe:fade-in-0 motion-safe:zoom-in-95 motion-safe:slide-in-from-bottom-4 motion-safe:duration-700">
    <img
      alt="ФИНКИ Хаб"
      className="h-16 w-16 object-contain drop-shadow-sm"
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
        {SUGGESTIONS.map((s, i) => (
          <button
            aria-label={s.replaceAll(NON_BREAKING_SPACE, ' ')}
            className={`${CHIP} ${STAGGER[i] ?? ''}`}
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

export const Thread = ({
  activeError,
  activeStatus,
  messages,
  onPickSuggestion,
  onRetry,
  renderActions,
  status,
}: ThreadProps) => {
  const lastAssistantId = messages.findLast((m) => m.role === 'assistant')?.id;
  const streaming = status === 'streaming' || status === 'submitted';
  const lastMessage = messages.at(-1);
  const awaitingReply =
    streaming &&
    (lastMessage?.role !== 'assistant' ||
      (activeStatus === undefined &&
        !hasText(lastMessage) &&
        !hasReasoning(lastMessage)));

  return (
    <Conversation className="flex-1">
      <ConversationContent>
        {messages.length === 0 ? (
          <Welcome onPick={onPickSuggestion} />
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
            {messages.map((m) => {
              if (
                m.role === 'assistant' &&
                m.id === lastAssistantId &&
                streaming &&
                activeStatus === undefined &&
                !hasText(m) &&
                !hasReasoning(m)
              ) {
                return null;
              }

              if (m.role === 'user') {
                return (
                  <Message
                    from="user"
                    key={m.id}
                  >
                    <MessageContent>
                      <MessageResponse>{joinText(m)}</MessageResponse>
                    </MessageContent>
                  </Message>
                );
              }

              const isLastAssistant = m.id === lastAssistantId;
              return (
                <AssistantMessage
                  actions={renderActions ? renderActions(m) : undefined}
                  errorPart={
                    isLastAssistant
                      ? (activeError ?? m.metadata?.error)
                      : m.metadata?.error
                  }
                  key={m.id}
                  message={m}
                  onRetry={isLastAssistant ? onRetry : undefined}
                  pending={isLastAssistant && streaming}
                  statusPart={
                    isLastAssistant && streaming ? activeStatus : undefined
                  }
                />
              );
            })}
            {awaitingReply ? (
              <Message from="assistant">
                <MessageContent>
                  <TypingIndicator />
                </MessageContent>
              </Message>
            ) : null}
          </div>
        )}
        {activeError && lastAssistantId === undefined ? (
          <div className="mx-auto w-full max-w-3xl">
            <MessageError
              errorPart={activeError}
              onRetry={onRetry}
            />
          </div>
        ) : null}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};

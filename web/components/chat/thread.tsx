import type { ReactNode } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

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
import { ElapsedTimer } from '@/components/chat/elapsed-timer';
import { AssistantMessage } from '@/components/chat/message';
import { TypingIndicator } from '@/components/chat/typing-indicator';
import { t } from '@/lib/i18n';
import { joinText } from '@/lib/message-parts';

export type ThreadProps = {
  activeError?: { code: string; message: string };
  activeStatus?: { label: string; tool?: string };
  messages: MyUIMessage[];
  onPickSuggestion?: (text: string) => void;
  onRetry?: () => void;
  renderActions?: (message: MyUIMessage) => ReactNode;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
  streamStartedAt?: null | number;
};

const SUGGESTIONS = [
  'Колку кредити ми требаат за да се запишам во наредна година?',
  'Колку изнесува школарината за една учебна година?',
  'Колку пати можам да полагам еден испит?',
  'Кои се условите за дипломирање?',
] as const;

const STAGGER = ['stagger-1', 'stagger-2', 'stagger-3', 'stagger-4'];

const CHIP =
  'group rounded-xl border border-border bg-card/60 px-4 py-3 text-left text-sm text-foreground/90 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40 hover:bg-card hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:fill-mode-both';

const Welcome = ({ onPick }: { onPick?: (text: string) => void }) => (
  <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center gap-6 py-12 text-center motion-safe:animate-in motion-safe:fade-in-0 motion-safe:zoom-in-95 motion-safe:slide-in-from-bottom-4 motion-safe:duration-700">
    <img
      alt="ФИНКИ Хаб"
      className="h-16 w-16 object-contain drop-shadow-sm"
      src="/logo.png"
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
            className={`${CHIP} ${STAGGER[i] ?? ''}`}
            key={s}
            onClick={() => {
              onPick(s);
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
  streamStartedAt,
}: ThreadProps) => {
  const lastAssistantId = messages.findLast((m) => m.role === 'assistant')?.id;
  const streaming = status === 'streaming' || status === 'submitted';
  const awaitingReply = streaming && messages.at(-1)?.role !== 'assistant';

  return (
    <Conversation className="flex-1">
      <ConversationContent>
        {messages.length === 0 ? (
          <Welcome onPick={onPickSuggestion} />
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
            {messages.map((m) => {
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
                  errorPart={isLastAssistant ? activeError : undefined}
                  key={m.id}
                  message={m}
                  onRetry={onRetry}
                  pending={isLastAssistant && streaming}
                  statusPart={
                    isLastAssistant && streaming ? activeStatus : undefined
                  }
                  streamStartedAt={
                    isLastAssistant ? streamStartedAt : undefined
                  }
                />
              );
            })}
            {awaitingReply ? (
              <Message from="assistant">
                <MessageContent>
                  <div className="flex items-center gap-2">
                    <TypingIndicator />
                    {typeof streamStartedAt === 'number' ? (
                      <ElapsedTimer startedAt={streamStartedAt} />
                    ) : null}
                  </div>
                </MessageContent>
              </Message>
            ) : null}
          </div>
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};

import type { ReactNode } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import { AssistantMessage } from '@/components/chat/message';
import { t } from '@/lib/i18n';

export type ThreadProps = {
  activeError?: { code: string; message: string };
  activeStatus?: { label: string; tool?: string };
  messages: MyUIMessage[];
  onRetry?: () => void;
  renderActions?: (message: MyUIMessage) => ReactNode;
  status: 'error' | 'ready' | 'streaming' | 'submitted';
};

const userText = (message: MyUIMessage): string =>
  message.parts
    .filter((p): p is { text: string; type: 'text' } => p.type === 'text')
    .map((p) => p.text)
    .join('');

export const Thread = ({
  activeError,
  activeStatus,
  messages,
  onRetry,
  renderActions,
  status,
}: ThreadProps) => {
  const lastAssistantId = messages.findLast((m) => m.role === 'assistant')?.id;
  const streaming = status === 'streaming' || status === 'submitted';

  return (
    <Conversation className="flex-1">
      <ConversationContent>
        {messages.length === 0 ? (
          <ConversationEmptyState
            description={t('thread.emptyDescription')}
            title={t('thread.emptyTitle')}
          />
        ) : (
          messages.map((m) => {
            if (m.role === 'user') {
              return (
                <Message
                  from="user"
                  key={m.id}
                >
                  <MessageContent>
                    <MessageResponse>{userText(m)}</MessageResponse>
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
                statusPart={
                  isLastAssistant && streaming ? activeStatus : undefined
                }
              />
            );
          })
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};

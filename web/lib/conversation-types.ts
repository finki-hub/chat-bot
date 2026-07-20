import type { MyUIMessage } from '@/lib/api-types';

export type ActiveConversationStream = {
  readonly id: string;
  readonly replacementMessageId: null | string;
};

export type ChatConversationHistory = {
  readonly conversation: ConversationRow & {
    readonly activeStream: ActiveConversationStream | null;
  };
  readonly messages: readonly MyUIMessage[];
};

export type ConversationRow = {
  readonly id: string;
  readonly model: null | string;
  readonly title: string;
};

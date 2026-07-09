import type { MyUIMessage } from '@/lib/api-types';

export type ChatConversationHistory = {
  readonly conversation: ConversationRow;
  readonly messages: readonly MyUIMessage[];
};

export type ConversationRow = {
  readonly id: string;
  readonly model: null | string;
  readonly title: string;
};

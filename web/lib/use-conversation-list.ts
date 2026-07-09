'use client';

import { useCallback, useEffect, useState } from 'react';

import type { ConversationRow } from '@/lib/conversation-types';

import { fireAndForget } from '@/lib/async';
import { listChatConversations } from '@/lib/transport';

export const useConversationList = () => {
  const [conversations, setConversations] = useState<ConversationRow[]>([]);

  const refreshConversations = useCallback(async () => {
    setConversations(await listChatConversations());
  }, []);

  useEffect(() => {
    fireAndForget(refreshConversations());
  }, [refreshConversations]);

  return { conversations, refreshConversations };
};

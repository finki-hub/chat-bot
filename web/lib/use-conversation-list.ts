'use client';

import { useCallback, useEffect, useState } from 'react';

import { fireAndForget } from '@/lib/async';
import { type ConversationRow, listConversations } from '@/lib/db';

export const useConversationList = () => {
  const [conversations, setConversations] = useState<ConversationRow[]>([]);

  const refreshConversations = useCallback(async () => {
    setConversations(await listConversations());
  }, []);

  useEffect(() => {
    fireAndForget(refreshConversations());
  }, [refreshConversations]);

  return { conversations, refreshConversations };
};

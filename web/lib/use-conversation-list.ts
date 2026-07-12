'use client';

import { useCallback, useEffect, useState } from 'react';

import type { ConversationRow } from '@/lib/conversation-types';

import { fireAndForget } from '@/lib/async';
import {
  ChatConversationRequestError,
  listChatConversations,
} from '@/lib/transport';

export const useConversationList = () => {
  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);

  const refreshConversations = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      setConversations(await listChatConversations());
    } catch (error_) {
      if (
        error_ instanceof ChatConversationRequestError ||
        error_ instanceof TypeError
      ) {
        setError(true);
        return;
      }
      throw error_;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fireAndForget(refreshConversations());
  }, [refreshConversations]);

  return { conversations, error, loading, refreshConversations };
};

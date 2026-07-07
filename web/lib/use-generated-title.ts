'use client';

import { type RefObject, useCallback, useState } from 'react';

import type { MyUIMessage } from '@/lib/api-types';

import { fireAndForget } from '@/lib/async';
import { generateChatTitle } from '@/lib/chat-title';
import {
  type ConversationRow,
  loadMessages,
  type MessageRow,
  renameConversationIfTitle,
} from '@/lib/db';

type UseGeneratedTitleOptions = {
  readonly conversations: readonly ConversationRow[];
  readonly modelRef: RefObject<string>;
  readonly refreshConversations: () => Promise<void>;
};

const fromRow = (row: MessageRow): MyUIMessage => ({
  id: row.id,
  metadata: row.metadata,
  parts: row.parts,
  role: row.role,
});

export const useGeneratedTitle = ({
  conversations,
  modelRef,
  refreshConversations,
}: UseGeneratedTitleOptions) => {
  const [generatingTitleId, setGeneratingTitleId] = useState<null | string>(
    null,
  );

  const applyGeneratedTitle = useCallback(
    async (
      id: string,
      titleMessages: readonly MyUIMessage[],
      expectedTitle: string,
    ): Promise<void> => {
      setGeneratingTitleId(id);
      try {
        const title = await generateChatTitle({
          messages: titleMessages,
          queryTransformModel: modelRef.current,
        });

        if (title === null) {
          return;
        }

        await renameConversationIfTitle(id, expectedTitle, title);
        await refreshConversations();
      } finally {
        setGeneratingTitleId((current) => (current === id ? null : current));
      }
    },
    [modelRef, refreshConversations],
  );

  const handleGenerateTitle = useCallback(
    (id: string) => {
      const expectedTitle = conversations.find((c) => c.id === id)?.title;
      if (expectedTitle === undefined) {
        return;
      }

      const run = async (): Promise<void> => {
        const rows = await loadMessages(id);
        if (rows.length === 0) {
          return;
        }
        await applyGeneratedTitle(id, rows.map(fromRow), expectedTitle);
      };

      fireAndForget(run());
    },
    [applyGeneratedTitle, conversations],
  );

  return { applyGeneratedTitle, generatingTitleId, handleGenerateTitle };
};
